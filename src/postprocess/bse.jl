@doc raw"""
Hermitian resonant (Tamm–Dancoff) singlet BSE:
```math
H_{tt'}=D_tδ_{tt'}+s_t(2K^x_{tt'}-K^d_{tt'})s_{t'},\quad
s_t=\sqrt{w_k(f_{vk}-f_{ck})/f_{full}}.
```
Negative occupation differences require a non-Hermitian nonequilibrium treatment
and are rejected. The kernels supplied here do not include k weights or populations.
"""
function assemble_bse(D,Kx,Kd,weights;hermitian_tolerance=1e-8)
    n = length(D)
    size(Kx) == size(Kd) == (n,n) || throw(DimensionMismatch("BSE kernel size mismatch."))
    length(weights) == n || throw(DimensionMismatch("BSE weight size mismatch."))
    all(≥(0),weights) || throw(ArgumentError("BSE square-root weights must be nonnegative."))
    H = Diagonal(D)+Diagonal(weights)*(2Kx-Kd)*Diagonal(weights)
    norm(H-H') ≤ hermitian_tolerance*max(one(real(eltype(H))),norm(H)) || error(
        "BSE kernel is not Hermitian; check q mapping, screening and occupations.")
    H
end

@doc raw"""
    compute_bse(data; valence, conduction, screening, Ecut_exchange, scissor=0, ...)

Dense singlet Tamm–Dancoff BSE on a full, serial CPU k mesh. `screening(q)` must
return a **static**, finite-q `compute_rpa` result at the requested reduced q,
including a justified Coulomb singularity treatment. No GW is performed.
```math
K^x_{vc k,v'c' k'}=Ω^{-1}\sum_{G≠0}ρ_{vc k}(G)^*v(G)ρ_{v'c' k'}(G),
\quad K^d_{vc k,v'c' k'}=Ω^{-1}ρ_{vv'}^\dagger W(k'-k,0)ρ_{cc'}.
```
The scissor changes transition energies, not orbitals or dipoles. Partially
occupied states use square-root occupation differences in both kernels and
oscillators. This static, intermanifold reference approximation omits dynamical
screening, resonant/antiresonant coupling and population inversion. Screening
cutoff, singularity integral, k grid and transition window require convergence.
"""
@timing function compute_bse(data;valence,conduction,screening,Ecut_exchange,scissor=0,
                             exchange_kernel=BareCoulomb(),max_matrix_bytes=512*1024^2)
    basis = data.basis
    _check_optical_basis(basis)
    mpi_nprocs(basis.comm_kpts) == 1 || error("Reference BSE is serial.")
    basis.model.spin_polarization == :none || error("Reference BSE supports spin singlets only.")
    isempty(intersect(valence,conduction)) || throw(ArgumentError("BSE manifolds overlap."))
    isempty(valence) || isempty(conduction) ? throw(ArgumentError("BSE manifolds are empty.")) : nothing
    ε,f = data.eigenvalues,data.occupation
    all(εk -> all(n -> 1 ≤ n ≤ length(εk),vcat(collect(valence),collect(conduction))),ε) ||
        throw(ArgumentError("BSE band index is out of range."))
    transitions = [(;v,c,ik) for ik in eachindex(basis.kpoints)
                   for v in valence for c in conduction]
    T = promote_type(eltype(basis),typeof(scissor))
    _check_response_memory(length(transitions),[0],T,max_matrix_bytes;arrays=6)
    D = [ε[t.ik][t.c]-ε[t.ik][t.v]+scissor for t in transitions]
    all(>(0),D) || throw(ArgumentError("BSE transition energies must be positive."))
    difference = [(f[t.ik][t.v]-f[t.ik][t.c])/filled_occupation(basis.model)
                  for t in transitions]
    all(≥(0),difference) || error("Population inversion requires a non-Hermitian BSE.")
    weights = [sqrt(basis.kweights[t.ik]*difference[i]) for (i,t) in enumerate(transitions)]
    u = [[ifft(basis,kpt,ψnk) for ψnk in eachcol(data.ψ[ik])]
         for (ik,kpt) in enumerate(basis.kpoints)]
    q0 = zero(Vec3{eltype(basis)})
    Gx = _response_vectors(basis,q0,Ecut_exchange)
    indices_x = _response_indices(basis,Gx)
    vx = vec(eval_kernel_fourier(exchange_kernel,basis,q0))[indices_x]
    head = findfirst(iszero,Gx)
    isnothing(head) || (vx[head] = 0)
    all(isfinite,vx) || error("Bare exchange kernel contains an untreated singularity.")
    P = [_pair_density_fourier(basis,u[t.ik][t.v],u[t.ik][t.c],zero(Vec3{Int}),indices_x)
         for t in transitions]
    Ω = basis.model.unit_cell_volume
    Kx = [dot(p,vx.*pprime)/Ω for p in P,pprime in P]
    Kd = zeros(Complex{T},length(transitions),length(transitions))
    for ik in eachindex(basis.kpoints),ikp in eachindex(basis.kpoints)
        k,kp = basis.kpoints[ik],basis.kpoints[ikp]
        q = normalize_kpoint_coordinate(kp.coordinate-k.coordinate)
        mapped,ΔG = find_equivalent_kpt(basis,k.coordinate+q,k.spin)
        mapped == ikp || error("Inconsistent BSE k-point mapping.")
        response = screening(q)
        response.optical_limit && error("Optically scaled W is not a finite-q BSE interaction.")
        response.q ≈ q || error("Screening returned a different q.")
        length(response.ω) == 1 && iszero(only(response.ω)) && iszero(response.η) ||
            error("BSE needs static W at ω=0, η=0.")
        response.basis.model.lattice ≈ basis.model.lattice || error("Screening lattice mismatch.")
        W = response.W[:,:,1]
        norm(W-W') ≤ 1e-8*max(one(T),norm(W)) || error("Static W is not Hermitian.")
        indices = _response_indices(basis,response.Gs)
        left = findall(t -> t.ik == ik,transitions)
        right = findall(t -> t.ik == ikp,transitions)
        for i in left,j in right
            t,tprime = transitions[i],transitions[j]
            ρvv = _pair_density_fourier(basis,u[ik][t.v],u[ikp][tprime.v],ΔG,indices)
            ρcc = _pair_density_fourier(basis,u[ik][t.c],u[ikp][tprime.c],ΔG,indices)
            Kd[i,j] = dot(ρvv,W*ρcc)/Ω
        end
    end
    H = assemble_bse(D,Kx,Kd,weights)
    F = eigen(Hermitian(H))
    (;basis,transitions,D,Kx,Kd,H,weights,population_factors=difference,scissor,
     energies=F.values,A=F.vectors,binding_energies=minimum(D).-F.values)
end

@doc raw"""
Optical dielectric tensor from BSE eigenvectors, including resonant and causal
antiresonant poles. With ``d_{λα}=\sum_t s_t A_{tλ} v^α_{vc}/Δε_{KS}``:
```math
ε_{αβ}=δ_{αβ}+\frac{8π}{Ω}\sum_λ\left[
\frac{d_{λα}d^*_{λβ}}{E_λ-ω-iη}+\frac{d^*_{λα}d_{λβ}}{E_λ+ω+iη}\right].
```
The factor two is the singlet spin degeneracy. Return dipoles, directional
oscillator strengths `2Eλ|dλα|²` and the spectrum. Negative excitation energies
signal an unstable reference and cannot be used in this spectral formula.
"""
function compute_bse_dielectric(data,bse,ω;η,velocity=compute_velocity(data))
    η > 0 || throw(ArgumentError("η must be positive."))
    all(>(0),bse.energies) || error("BSE has nonpositive excitation energies.")
    T = real(eltype(bse.A))
    dipoles = zeros(Complex{T},length(bse.energies),3)
    for λ in eachindex(bse.energies),α = 1:3,(i,t) in enumerate(bse.transitions)
        Δ = data.eigenvalues[t.ik][t.c]-data.eigenvalues[t.ik][t.v]
        abs(Δ) > 1e-10 || error("An unshifted BSE transition is degenerate.")
        dipoles[λ,α] += bse.weights[i]*bse.A[i,λ]*velocity[t.ik][t.v,t.c,α]/Δ
    end
    ε = zeros(Complex{T},3,3,length(ω))
    for α = 1:3
        ε[α,α,:] .= 1
    end
    for iω in eachindex(ω),λ in eachindex(bse.energies),α = 1:3,β = 1:3
        z = ω[iω]+im*η
        dα,dβ = dipoles[λ,α],dipoles[λ,β]
        ε[α,β,iω] += 8π/bse.basis.model.unit_cell_volume *
            (dα*conj(dβ)/(bse.energies[λ]-z)+conj(dα)*dβ/(bse.energies[λ]+z))
    end
    (;ω,ε,dipoles,oscillator_strengths=2bse.energies.*abs2.(dipoles))
end

function _exciton_amplitudes(bse,λ)
    A = sqrt.(bse.population_factors).*bse.A[:,λ]
    norm(A) > 0 || error("This state has zero electron–hole spectral weight.")
    A/norm(A)
end

"""
Normalized periodic electron/hole marginals on the FFT grid. For a partially
occupied reference, use normalized `sqrt(fv-fc)*A` transition amplitudes. These
are descriptors of the static BSE model, not a nonequilibrium density matrix.
"""
function compute_exciton_densities(data,bse,λ)
    basis = data.basis
    A = _exciton_amplitudes(bse,λ)
    ρelectron = zeros(eltype(basis),basis.fft_size)
    ρhole = zero(ρelectron)
    for (ik,v) in unique([(t.ik,t.v) for t in bse.transitions])
        field = zeros(complex(eltype(basis)),basis.fft_size)
        for (i,t) in enumerate(bse.transitions)
            (t.ik == ik && t.v == v) || continue
            field .+= A[i].*ifft(basis,basis.kpoints[ik],data.ψ[ik][:,t.c])
        end
        ρelectron .+= abs2.(field)
    end
    for (ik,c) in unique([(t.ik,t.c) for t in bse.transitions])
        field = zeros(complex(eltype(basis)),basis.fft_size)
        for (i,t) in enumerate(bse.transitions)
            (t.ik == ik && t.c == c) || continue
            field .+= A[i].*conj.(ifft(basis,basis.kpoints[ik],data.ψ[ik][:,t.v]))
        end
        ρhole .+= abs2.(field)
    end
    (;ρelectron,ρhole)
end

@doc raw"""
Joint and conditional fragment probability for periodic real-space masks in [0,1].
Evaluate ``⟨A|P_h⊗P_e|A⟩``, retaining coherence within each k block. Returns the
electron fraction in `electron_mask` conditioned on the hole in `hole_mask`.
This is a partition-dependent CT descriptor, not a product of marginal charges.
"""
function compute_exciton_character(data,bse,λ;electron_mask,hole_mask)
    basis = data.basis
    for mask in (electron_mask,hole_mask)
        size(mask) == basis.fft_size || throw(DimensionMismatch("Exciton mask grid mismatch."))
        all(x -> 0 ≤ x ≤ 1,mask) || throw(ArgumentError("Exciton masks must lie in [0,1]."))
    end
    A = _exciton_amplitudes(bse,λ)
    joint = zero(eltype(basis))
    for ik in eachindex(basis.kpoints)
        indices = findall(t -> t.ik == ik,bse.transitions)
        u = [ifft(basis,basis.kpoints[ik],ψn) for ψn in eachcol(data.ψ[ik])]
        for i in indices,j in indices
            t,tprime = bse.transitions[i],bse.transitions[j]
            e = sum(conj.(u[t.c]).*electron_mask.*u[tprime.c])*basis.dvol
            h = sum(u[t.v].*hole_mask.*conj.(u[tprime.v]))*basis.dvol
            joint += real(conj(A[i])*A[j]*e*h)
        end
    end
    densities = compute_exciton_densities(data,bse,λ)
    hole_probability = sum(hole_mask.*densities.ρhole)*basis.dvol
    hole_probability > 0 || error("The selected hole region has zero probability.")
    (;joint,hole_probability,conditional_electron_fraction=joint/hole_probability)
end

@doc raw"""
Isotropic **3D** Wannier–Mott estimate, ``μ=m_em_h/(m_e+m_h)``,
``E_B=μ/(2ε_∞²)`` (Ha), ``a_B=ε_∞/μ`` (Bohr). Masses are in electron masses.
This hydrogenic estimate is not a monolayer or interfacial CT exciton model.
"""
function wannier_mott(m_e,m_h,ε∞)
    all(x -> isfinite(x) && x > 0,(m_e,m_h,ε∞)) ||
        throw(ArgumentError("Masses and dielectric constant must be finite and positive."))
    μ = m_e*m_h/(m_e+m_h)
    (;reduced_mass=μ,binding_energy=μ/(2ε∞^2),radius=ε∞/μ)
end
