# Dense, serial CPU reference response. Pair densities live on the FFT cube,
# not on a k-point's orbital sphere.

function _response_indices(basis,Gs)
    lookup = Dict(G=>i for (i,G) in enumerate(G_vectors(basis)))
    length(unique(Gs)) == length(Gs) || throw(ArgumentError("Response G vectors must be unique."))
    all(G -> haskey(lookup,G),Gs) || error("Response G vectors are outside the FFT grid.")
    [lookup[G] for G in Gs]
end

function _response_vectors(basis,q,Ecut_response)
    Ecut_response > 0 || throw(ArgumentError("Ecut_response must be positive."))
    [G for G in G_vectors(basis)
     if norm2(basis.model.recip_lattice*(G+q))/2 ≤ Ecut_response]
end

function _check_response_memory(n,frequencies,T,max_matrix_bytes;arrays=1)
    bytes = big(n)^2*length(frequencies)*sizeof(Complex{T})*arrays
    bytes ≤ max_matrix_bytes || error(
        "Dense reference response needs at least $bytes bytes; lower Ecut_response " *
        "or split the frequency grid (limit $max_matrix_bytes bytes).")
end

# Fourier coefficient of u_nk* u_m,k+q, with the reciprocal relabelling used by EXX.
function _pair_density_fourier(basis,u,v,ΔG,indices)
    product = conj.(u).*v
    iszero(ΔG) || (product .*= map(r -> cis2pi(dot(ΔG, r)), r_vectors(basis)))
    sqrt(basis.model.unit_cell_volume) .* vec(fft(basis, product))[indices]
end

function _scissored_eigenvalues(eigenvalues,scissor,n_valence)
    iszero(scissor) && return eigenvalues
    isnothing(n_valence) && throw(ArgumentError("A scissor requires n_valence."))
    all(ε -> 0 < n_valence < length(ε),eigenvalues) ||
        throw(ArgumentError("Both scissor manifolds must be present."))
    [[ε[n]+(n > n_valence ? scissor : zero(scissor)) for n in eachindex(ε)]
     for ε in eigenvalues]
end

@doc raw"""
    compute_χ0_dynamic(data, q, ω; η, Ecut_response, ...)

Adler–Wiser independent-particle response in the dimensionless pair-density basis:
```math
χ^0_{GG'}(q,ω)=\frac1Ω\sum_{knm}w_k
\frac{f_{nk}-f_{m,k+q}}{ω+ε_{nk}-ε_{m,k+q}+iη}
ρ_{nm}(G)ρ_{nm}(G')^*.
```
`q` is reduced; it must map the full k mesh onto itself. The implementation is
serial CPU, without symmetry reduction. `Gs` optionally specifies response G vectors
on the FFT cube; otherwise use the explicit `Ecut_response` (Ha).

`occupation_derivative` supplies static degenerate/intraband limits; it is essential
for metallic static screening. No thermal derivative is guessed from arbitrary
occupations. For `q=0`, optional Cartesian `optical_direction` replaces the head pair
density by `e⋅v_nm/(ε_m-ε_n)`, so head/wings are divided by q²/q analytically.
This optical limit is **interband only**; use finite q for free-carrier screening
and plasmons, or `compute_dielectric` for an explicit optical Drude term.
"""
@timing function compute_χ0_dynamic(data,q,ω;η,Ecut_response,
                                    Gs=nothing,scissor=0,n_valence=nothing,
                                    occupation_derivative=nothing,optical_direction=nothing,
                                    max_matrix_bytes=512*1024^2,degeneracy_tol=1e-10)
    basis = data.basis
    _check_optical_basis(basis)
    mpi_nprocs(basis.comm_kpts) == 1 || error("Reference dynamical response is serial.")
    η ≥ 0 || throw(ArgumentError("η must be nonnegative."))
    all(isfinite,ω) && all(ω .≥ 0) || throw(ArgumentError("ω must be finite and nonnegative."))
    q = Vec3{eltype(basis)}(q)
    Gs = isnothing(Gs) ? _response_vectors(basis,q,Ecut_response) : Vec3{Int}.(Gs)
    isempty(Gs) && throw(ArgumentError("The response basis is empty."))
    indices = _response_indices(basis,Gs)
    T = promote_type(eltype(basis),eltype(ω),typeof(η),typeof(scissor))
    _check_response_memory(length(Gs),ω,T,max_matrix_bytes)
    χ0 = zeros(Complex{T},length(Gs),length(Gs),length(ω))
    ε = _scissored_eigenvalues(data.eigenvalues,scissor,n_valence)
    f = data.occupation
    length(f) == length(ε) == length(basis.kpoints) ||
        throw(DimensionMismatch("One band/occupation array is needed per k point."))
    optical_limit = !isnothing(optical_direction)
    if optical_limit
        iszero(q) || throw(ArgumentError("optical_direction is only valid at q=0."))
        norm(optical_direction) > 0 || throw(ArgumentError("Optical direction must be nonzero."))
    end
    direction = optical_limit ? normalize(optical_direction) : nothing
    velocity = optical_limit ? compute_velocity(data) : nothing
    head = findfirst(iszero,Gs)
    optical_limit && isnothing(head) && error("The optical limit needs G=0.")
    for (ik,kpt) in enumerate(basis.kpoints)
        ikp,ΔG = find_equivalent_kpt(basis,kpt.coordinate+q,kpt.spin)
        u = [ifft(basis, kpt, ψn) for ψn in eachcol(data.ψ[ik])]
        up = ik == ikp ? u :
             [ifft(basis, basis.kpoints[ikp], ψm) for ψm in eachcol(data.ψ[ikp])]
        for n in eachindex(ε[ik]),m in eachindex(ε[ikp])
            Δ = ε[ik][n]-ε[ikp][m]
            δf = f[ik][n]-f[ikp][m]
            degenerate_static = abs(Δ) ≤ degeneracy_tol && abs(δf) ≤ degeneracy_tol &&
                                !isnothing(occupation_derivative)
            iszero(δf) && !degenerate_static && continue
            ρnm = _pair_density_fourier(basis, u[n], up[m], ΔG, indices)
            if optical_limit
                ρnm[head] = if abs(Δ) ≤ degeneracy_tol
                    zero(eltype(ρnm))
                else
                    # The scissor rescales v and Δ equally: the dipole is unchanged.
                    ΔKS = data.eigenvalues[ikp][m]-data.eigenvalues[ik][n]
                    sum(direction[α]*velocity[ik][n,m,α] for α = 1:3)/ΔKS
                end
            end
            for iω in eachindex(ω)
                z = ω[iω]+im*η
                if iszero(z) && degenerate_static
                    ratio = (occupation_derivative[ik][n]+occupation_derivative[ikp][m])/2
                else
                    iszero(δf) && continue
                    abs(z+Δ) > eps(T) || error("Response pole at η=0; choose positive η.")
                    ratio = δf/(z+Δ)
                end
                χ0[:,:,iω] .+= (basis.kweights[ik]/basis.model.unit_cell_volume*ratio) .*
                                (ρnm*ρnm')
            end
        end
    end
    (;basis,q,ω,η,Gs,χ0,optical_limit,scissor,n_valence)
end

@doc raw"""
Dense RPA screening: ``ε=I-vχ^0``, ``W=ε^{-1}v``. Return the dielectric matrix,
its inverse, screened interaction and macroscopic dielectric function.
The default omits the isolated q+G=0 singularity. **Converged BSE calculations
need a justified Brillouin-zone singularity integral**, e.g. `VoxelAverage` with
the corresponding extension; omitting one mesh point is not such a convergence test.
For an analytic optical limit the scaled Coulomb head is 4π; its W is scaled and
must not be used as a finite-q BSE interaction.
"""
@timing function compute_rpa(response::NamedTuple;
                             kernel=ReplaceSingularity(BareCoulomb(),0),
                             max_matrix_bytes=512*1024^2)
    (;basis,q,ω,Gs,χ0,optical_limit) = response
    T = real(eltype(χ0))
    _check_response_memory(length(Gs),ω,T,max_matrix_bytes;arrays=4)
    coefficients = vec(eval_kernel_fourier(kernel,basis,q))[_response_indices(basis,Gs)]
    head = findfirst(iszero,Gs)
    optical_limit && (coefficients[head] = 4T(π))
    all(isfinite,coefficients) || error("Choose an explicit Coulomb singularity treatment.")
    v = Diagonal(coefficients)
    ε = similar(χ0)
    εinverse = similar(χ0)
    W = similar(χ0)
    for iω in eachindex(ω)
        ε[:,:,iω] = I-v*χ0[:,:,iω]
        εinverse[:,:,iω] = inv(ε[:,:,iω])
        W[:,:,iω] = εinverse[:,:,iω]*v
    end
    εmacroscopic = isnothing(head) ? nothing : inv.(εinverse[head,head,:])
    (;response...,ε,εinverse,W,εmacroscopic,kernel)
end

function compute_rpa(data::NamedTuple,q,ω;
                     kernel=ReplaceSingularity(BareCoulomb(),0),
                     max_matrix_bytes=512*1024^2,kwargs...)
    response = compute_χ0_dynamic(data,q,ω;max_matrix_bytes,kwargs...)
    compute_rpa(response;kernel,max_matrix_bytes)
end

@doc raw"""
EELS with microscopic local fields: ``L(q,ω)=-\Im ε^{-1}_{00}(q,ω)``.
`response` is the output of `compute_rpa`; q=0 optical limits are interband only.
"""
function compute_eels(response::NamedTuple)
    get(response, :screening_voxel, false) && throw(ArgumentError(
        "A screened cell average is not a pointwise EELS response."))
    head = findfirst(iszero,response.Gs)
    isnothing(head) && error("EELS requires the response head G=0.")
    (;q=response.q,ω=response.ω,loss=-imag.(response.εinverse[head,head,:]))
end
