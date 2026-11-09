# Static long-wavelength screening, including a cell integral of the screened head.

@doc raw"""
Build the long-wavelength Adler–Wiser coefficients:
```math
χ^0(q)=K+\begin{pmatrix}q^T A q&q^T B\\B^\dagger q&C\end{pmatrix},\qquad
\begin{pmatrix}A&B\\B^\dagger&C\end{pmatrix}
=\frac1Ω\sum_{knm}'w_k\frac{f_n-f_m}{ε_n-ε_m}
\begin{pmatrix}d_{nm}\\ρ_{nm}\end{pmatrix}
\begin{pmatrix}d_{nm}\\ρ_{nm}\end{pmatrix}^\dagger.
```
Here `d_nm = v_nm/(ε_m_KS-ε_n_KS)` is Cartesian, `ρ_nm` contains G≠0,
and the primed sum excludes degenerate transitions. `K` is the static degenerate
term, with `(f_n-f_m)/(ε_n-ε_m)` replaced by the explicit occupation derivative.
It includes carrier compressibility. No derivative is inferred from fixed vectors.
"""
function _rpa_optical_expansion(data; Ecut_response, Gs=nothing, scissor=0, n_valence=nothing,
                                occupation_derivative=nothing, degeneracy_tol=1e-10,
                                occupation_tol=1e-10, max_matrix_bytes=512*1024^2)
    basis = data.basis
    _check_optical_basis(basis)
    mpi_nprocs(basis.comm_kpts) == 1 || error("Reference screening is serial.")
    q0 = zero(Vec3{eltype(basis)})
    Gs = isnothing(Gs) ? _response_vectors(basis, q0, Ecut_response) : Vec3{Int}.(Gs)
    head = findfirst(iszero, Gs)
    isnothing(head) && error("The optical expansion requires G=0.")
    body = filter(!=(head), eachindex(Gs))
    indices = _response_indices(basis, Gs[body])
    T = promote_type(eltype(basis), typeof(scissor))
    _check_response_memory(length(Gs)+2, [0], T, max_matrix_bytes; arrays=6)
    f = data.occupation
    if isnothing(occupation_derivative)
        full = filled_occupation(basis.model)
        full_or_empty(x) = abs(x) ≤ occupation_tol || abs(x-full) ≤ occupation_tol
        all(fk -> all(full_or_empty, fk), f) ||
            error("Partially occupied screening requires an explicit occupation derivative.")
    else
        length(occupation_derivative) == length(f) &&
            all(length(d) == length(fk) for (d, fk) in zip(occupation_derivative, f)) ||
            throw(DimensionMismatch("Occupation derivative shape mismatch."))
        all(d -> all(x -> isfinite(x) && x ≤ occupation_tol, d), occupation_derivative) ||
            error("Static carrier screening requires finite nonpositive derivatives.")
    end
    ε = _scissored_eigenvalues(data.eigenvalues, scissor, n_valence)
    velocity = compute_velocity(data)
    M = zeros(Complex{T}, length(body)+3, length(body)+3)
    K = zeros(Complex{T}, length(body)+1, length(body)+1)
    for (ik, kpt) in enumerate(basis.kpoints)
        # One k block at a time: do not cache real-space orbitals on the entire mesh.
        u = isempty(body) ? nothing : [ifft(basis, kpt, ψn) for ψn in eachcol(data.ψ[ik])]
        for n in eachindex(f[ik]), m in eachindex(f[ik])
            Δ, δf = ε[ik][n]-ε[ik][m], f[ik][n]-f[ik][m]
            degenerate = abs(Δ) ≤ degeneracy_tol
            if degenerate && isnothing(occupation_derivative) && abs(δf) ≤ occupation_tol
                continue
            end
            !degenerate && iszero(δf) && continue
            ρnm = isempty(body) ? Complex{T}[] :
                _pair_density_fourier(basis, u[n], u[m], zero(Vec3{Int}), indices)
            weight = basis.kweights[ik]/basis.model.unit_cell_volume
            if degenerate
                isnothing(occupation_derivative) && error(
                    "Unequal populations in a degenerate subspace need a density matrix.")
                derivative = (occupation_derivative[ik][n]+occupation_derivative[ik][m])/2
                abs(δf-derivative*Δ) ≤ 100eps(T)*max(1, abs(f[ik][n]), abs(f[ik][m])) ||
                    error("Inconsistent populations in a degenerate subspace.")
                y = vcat(T(n==m), ρnm)
                K .+= (weight*derivative).*(y*y')
            else
                ΔKS = data.eigenvalues[ik][m]-data.eigenvalues[ik][n]
                abs(ΔKS) > degeneracy_tol && -ΔKS*Δ > 0 ||
                    error("A scissor reversed or closed a screening transition.")
                x = vcat(velocity[ik][n, m, :]/ΔKS, ρnm)
                M .+= (weight*δf/Δ).*(x*x')
            end
        end
    end
    (; basis, Gs, head, body, A=M[1:3, 1:3], B=M[1:3, 4:end], C=M[4:end, 4:end], K,
      scissor, n_valence)
end

function _integrate_screening_voxel(expansion, kernel, basis, n)
    error("Screened optical cell integration requires loading FastGaussQuadrature.")
end

@doc raw"""
    compute_rpa_screening_voxel(data; Ecut_response, kernel=BareCoulomb(), ...)

Static **screened** interaction averaged over the central reciprocal cell:
```math
\bar W(0)=|V_q|^{-1}\int_{V_q}[v(q)^{-1}-χ^0(q)]^{-1}\,dq.
```
Use the long-wavelength expansion from optical velocities, including local fields
and an explicitly supplied carrier occupation derivative. Bulk insulators integrate
the anisotropic `4π/(qᵀεq)` head by a surface reduction. Sheets subtract the analytic
`4πR/|q|` singularity. A finite carrier compressibility makes the head integrable by
ordinary quadrature. This requires `FastGaussQuadrature` and an even quadrature order.

This is a long-wavelength cell model, not exact finite-q screening throughout a
coarse cell. Converge the k/q mesh and quadrature, especially relative to carrier
Fermi wavevectors. Return an unscaled `W` suitable for BSE `screening(q=0)`;
use finite-q `compute_rpa` elsewhere. The result is not a pointwise EELS response.
Supported kernels are bare 3D Coulomb and half-height slab-truncated Coulomb.
`dielectric_tensor` is returned only for insulating bulk; metallic static
screening and isolated sheets do not have that finite bulk dielectric limit.
"""
function compute_rpa_screening_voxel(data; Ecut_response, kernel=BareCoulomb(),
                                      n_quadrature_points=12, kwargs...)
    kernel isa Union{BareCoulomb,SlabTruncatedCoulomb} ||
        error("Screened cell integration supports bare bulk or slab Coulomb kernels.")
    expansion = _rpa_optical_expansion(data; Ecut_response, kwargs...)
    integrated = _integrate_screening_voxel(expansion, kernel, data.basis,
                                            n_quadrature_points)
    T = real(eltype(expansion.A))
    W = zeros(Complex{T}, length(expansion.Gs), length(expansion.Gs), 1)
    order = vcat(expansion.head, expansion.body)
    W[order, order, 1] = integrated.W
    (; basis=data.basis, q=zero(Vec3{T}), ω=[zero(T)], η=zero(T), Gs=expansion.Gs, W,
      optical_limit=false, screening_voxel=true, kernel, n_quadrature_points,
      integrated.dielectric_tensor, integrated.compressibility)
end
