# CPU reference implementation. All energies, frequencies and lengths are atomic units.

function _check_optical_basis(basis)
    basis.architecture isa CPU || error("Reference optics requires a CPU basis.")
    basis.model.spin_polarization == :full && error(
        "Reference optics does not yet support spinor orbitals.")
    length(basis.symmetries) == 1 || error(
        "Reference optics requires symmetries=false and an unreduced k grid.")
end

# Shift k, keeping exactly the same plane waves and FFT mapping. Rebuilding a cutoff
# sphere here would differentiate a discontinuous change of discretisation.
function _shift_kpoint(basis, kpt, δk_cart)
    coordinate = kpt.coordinate + basis.model.inv_recip_lattice * δk_cart
    Kpoint(kpt.spin, coordinate, kpt.G_vectors, kpt.mapping,
           kpt.mapping_inv, kpt.mapping_device)
end

function _optical_kmatrix(builder::Kinetic, term, basis, kpt, ψk; kwargs...)
    ψk' * Diagonal(kinetic_energy(builder, basis.Ecut,
                                  Gplusk_vectors_cart(basis, kpt))) * ψk
end

function _optical_kmatrix(::AtomicNonlocal, term, basis, kpt, ψk; kwargs...)
    groups = [g for g in basis.model.atom_groups
              if basis.model.atoms[first(g)] isa ElementPsp]
    isempty(groups) && return zero(ψk' * ψk)
    psps = [basis.model.atoms[first(g)].psp for g in groups]
    positions = [basis.model.positions[g] for g in groups]
    P = build_projection_vectors(basis, kpt, psps, positions)
    D = build_projection_coefficients(basis, psps, positions)
    ψP = ψk' * P
    ψP * D * ψP'
end

function _optical_kmatrix(::Hubbard, term::TermHubbard, basis, kpt, ψk;
                          hubbard_n, kwargs...)
    isempty(term.U) && return zero(ψk' * ψk)
    isnothing(hubbard_n) && error("Hubbard velocities require hubbard_n.")
    (; orbitals, labels) = _atomic_orbitals_nonorthogonal(basis; kpoints=[kpt])
    P = ortho_lowdin(only(orbitals))
    projectors = reduce(hcat, [only(extract_manifold(m, [P], labels).manifold_projectors)
                              for m in term.manifolds])
    D = hubbard_potential_coefficients(term, basis, hubbard_n)[kpt.spin]
    ψP = ψk' * projectors
    ψP * D * ψP'
end

function _optical_kmatrix(builder, term, basis, kpt, ψk; kwargs...)
    # Local multiplication and constant energies have no explicit k dependence.
    zero(ψk' * ψk)
end

@doc raw"""
    compute_velocity(band_data)

Cartesian velocity matrices `velocity[ik][n,m,α]`, in atomic units:
```math
v^α_{nm}(k) = \langle ψ_{nk}|\partial_{k_α} H_k|ψ_{mk}\rangle.
```
ForwardDiff differentiates the kinetic, nonlocal pseudopotential and Löwdin Hubbard
projector code at fixed density, Hubbard occupations, orbitals and plane waves. Local
potentials have zero explicit derivative. Exact exchange, meta-GGA, spinors, GPU and
symmetry reduction are deliberately unsupported in this reference implementation.
Empty bands must be converged separately, e.g. with `compute_bands` on a full k grid.
"""
@timing function compute_velocity(data::NamedTuple)
    basis = data.basis
    _check_optical_basis(basis)
    hubbard_n = get(data, :hubbard_n, nothing)
    for (builder, term) in zip(basis.model.term_types, basis.terms)
        isnothing(term) && continue
        builder isa Union{Kinetic,AtomicNonlocal,Hubbard} && continue
        term isa TermExactExchange && error("Exact-exchange velocities are not supported.")
        ops = ene_ops(term, basis, data.ψ, data.occupation;
                      ρ=data.ρ, τ=get(data, :τ, nothing), hubbard_n,
                      eigenvalues=data.eigenvalues, εF=get(data, :εF, nothing)).ops
        all(op -> op isa Union{NoopOperator,RealSpaceMultiplication}, ops) || error(
            "Optical k derivative is not implemented for $(typeof(builder)).")
    end
    map(enumerate(basis.kpoints)) do (ik, kpt)
        ψk = data.ψ[ik]
        function H(δk)
            shifted = _shift_kpoint(basis, kpt, δk)
            sum(zip(basis.model.term_types, basis.terms)) do (builder, term)
                _optical_kmatrix(builder, term, basis, shifted, ψk; hubbard_n)
            end
        end
        # Splitting real and imaginary parts keeps ForwardDiff's output real-valued.
        J = ForwardDiff.jacobian(zeros(eltype(basis), 3)) do δk
            h = H(δk)
            vcat(vec(real.(h)), vec(imag.(h)))
        end
        n = size(ψk, 2)
        reshape(J[1:n^2, :] + im * J[n^2+1:end, :], n, n, 3)
    end
end

@doc raw"""
    compute_drude_weight(basis, velocity, occupation_derivative)

Drude plasma-frequency tensor, with **explicit** `occupation_derivative[ik][n] = ∂f/∂ε`:
```math
ω^2_{p,αβ} = \frac{4π}{Ω}\sum_{kn} w_k (-f'_{nk}) v^α_{nn}v^β_{nn}.
```
The derivative is not inferred from arbitrary fixed occupation vectors. A nonzero
temperature or a separate Fermi-surface integration is needed for metals.
"""
function compute_drude_weight(basis, velocity, occupation_derivative)
    _check_optical_basis(basis)
    length(velocity) == length(occupation_derivative) == length(basis.kpoints) ||
        throw(DimensionMismatch("One velocity and derivative array is required per k point."))
    T = real(eltype(first(velocity)))
    ωp2 = zeros(T, 3, 3)
    for ik in eachindex(velocity), n in eachindex(occupation_derivative[ik]), α = 1:3, β = 1:3
        ωp2[α, β] -= (basis.kweights[ik] * occupation_derivative[ik][n]
                      * real(velocity[ik][n,n,α] * velocity[ik][n,n,β]))
    end
    4T(π) / basis.model.unit_cell_volume * mpi_sum(ωp2, basis.comm_kpts)
end

@doc raw"""
    compute_dielectric(basis, eigenvalues, occupation, velocity, ω; η, ...)

Independent-particle dielectric tensor `ε[α,β,iω]`, without local-field corrections:
```math
ε_{αβ}(ω)=δ_{αβ}-\frac{4π}{Ω}\sum_{k,n≠m}w_k
\frac{(f_{nk}-f_{mk})v^α_{nm}v^β_{mn}}
{(ε_{nk}-ε_{mk})^2(ω+ε_{nk}-ε_{mk}+iη)}
-\frac{ω^2_{p,αβ}}{ω(ω+iη)}.
```
Pass `occupation_derivative` to include the Drude term. A nonzero `scissor` requires
`n_valence`; intermanifold velocities are rescaled by `Δε_QP/Δε_KS`. The occupation
vectors already include spin degeneracy. Frequencies, broadening and scissor are Ha.
Degenerate states with equal occupations give zero interband contribution; unequal
occupations within a degenerate subspace require a separately specified density matrix.
For a slab, this is the supercell dielectric tensor and depends on the vacuum height.
"""
@timing function compute_dielectric(basis, eigenvalues, occupation, velocity, ω;
                                    η, scissor=0, n_valence=nothing,
                                    occupation_derivative=nothing,
                                    degeneracy_tol=1e-9)
    _check_optical_basis(basis)
    η > 0 || throw(ArgumentError("η must be positive."))
    all(isfinite, ω) && all(ω .≥ 0) || throw(ArgumentError("ω must be finite and nonnegative."))
    iszero(scissor) || !isnothing(n_valence) ||
        throw(ArgumentError("A scissor shift requires n_valence."))
    length(eigenvalues) == length(occupation) == length(velocity) ==
        length(basis.kpoints) || throw(DimensionMismatch("Inconsistent k-point arrays."))
    T = promote_type(eltype(basis), eltype(ω), typeof(η), typeof(scissor))
    ε = zeros(Complex{T}, 3, 3, length(ω))
    for α = 1:3
        ε[α,α,:] .= 1
    end
    χ = zero(ε)
    for ik in eachindex(eigenvalues)
        εk, fk, vk = eigenvalues[ik], occupation[ik], velocity[ik]
        length(εk) == length(fk) == size(vk,1) == size(vk,2) ||
            throw(DimensionMismatch("Inconsistent band arrays."))
        isnothing(n_valence) || 0 < n_valence < length(εk) ||
            throw(ArgumentError("n_valence must leave both valence and conduction bands."))
        for n in eachindex(εk), m in eachindex(εk)
            n == m && continue
            δf = fk[n] - fk[m]
            iszero(δf) && continue
            Δ = εk[n] - εk[m]
            abs(Δ) ≤ degeneracy_tol && abs(δf) ≤ sqrt(eps(T)) && continue
            abs(Δ) > degeneracy_tol || throw(ArgumentError(
                "Unequal occupations in a degenerate subspace at k point $ik."))
            shift(n) = !isnothing(n_valence) && n > n_valence ? scissor : zero(scissor)
            ΔQP = Δ + shift(n) - shift(m)
            ΔQP * Δ > 0 || throw(ArgumentError("The scissor closes or reverses a transition."))
            scale = ΔQP / Δ
            for α = 1:3, β = 1:3, iω in eachindex(ω)
                χ[α,β,iω] += (basis.kweights[ik] * δf * scale^2
                              * vk[n,m,α] * vk[m,n,β]
                              / (ΔQP^2 * (ω[iω] + ΔQP + im*η)))
            end
        end
    end
    ε .-= 4T(π) / basis.model.unit_cell_volume .* mpi_sum(χ, basis.comm_kpts)
    if !isnothing(occupation_derivative)
        ωp2 = compute_drude_weight(basis, velocity, occupation_derivative)
        if any(!iszero, ωp2)
            any(iszero, ω) && throw(ArgumentError("The Drude term is singular at ω=0."))
            for iω in eachindex(ω)
                ε[:,:,iω] .-= ωp2 / (ω[iω] * (ω[iω] + im*η))
            end
        end
    end
    (; ω, ε)
end

function compute_dielectric(data::NamedTuple, ω; velocity=compute_velocity(data), kwargs...)
    compute_dielectric(data.basis, data.eigenvalues, data.occupation, velocity, ω; kwargs...)
end

@doc raw"""
Principal refractive index of a scalar dielectric function: ``n+iκ=\sqrt{ε}``.
The branch has nonnegative `n`; negative `κ` represents optical gain.
"""
function refractive_index(ε::Number)
    index = sqrt(complex(ε))
    (; n=real(index), κ=imag(index))
end

@doc raw"""
Absorption coefficient (inverse Bohr): ``α(ω)=2ωκ(ω)/c`` in atomic units.
Apply to a scalar polarization or principal dielectric component, not elementwise to
a general anisotropic tensor. Negative absorption represents gain.
"""
absorption_coefficient(ω::Real, ε::Number) = 2ω * refractive_index(ε).κ / 137.035999084

"""Scalar energy loss function `-Im(1/ε)`. For local fields use `compute_eels`."""
loss_function(ε::Number) = -imag(inv(ε))

@doc raw"""
    fit_tauc(ω, α; window, exponent)

Least-squares line fit ``(αω)^{\mathrm{exponent}}=aω+b`` within the explicitly
chosen energy window. Return `gap=-b/a`, slope, intercept and R². Use exponent 2
for an allowed direct transition or 1/2 for an allowed indirect transition.
This numerical fit alone does not establish which transition mechanism is present.
"""
function fit_tauc(ω, α; window, exponent)
    length(ω) == length(α) || throw(DimensionMismatch("ω and α must have equal lengths."))
    exponent > 0 || throw(ArgumentError("The Tauc exponent must be positive."))
    indices = findall(i -> window[1] ≤ ω[i] ≤ window[2] && α[i] > 0,
                      eachindex(ω))
    length(indices) ≥ 3 || throw(ArgumentError("The Tauc window needs at least three points."))
    x = ω[indices]
    y = (α[indices] .* x).^exponent
    slope, intercept = hcat(x, ones(eltype(x), length(x))) \ y
    slope > 0 || throw(ArgumentError("The Tauc fit must have positive slope."))
    residual = sum(abs2, y .- (slope .* x .+ intercept))
    variance = sum(abs2, y .- sum(y)/length(y))
    (; gap=-intercept/slope, slope, intercept, r2=1-residual/variance,
     window, exponent, indices)
end
