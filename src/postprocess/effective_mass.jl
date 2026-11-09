@doc raw"""
    compute_effective_mass(data; band, ik=1, carrier=:electron)

Reference Cartesian effective-mass tensor of an **isolated, nondegenerate** band:
```math
(m^{-1})_{αβ}=\langle n|\partial_{k_α}\partial_{k_β}H|n\rangle+
2\Re\sum_{m≠n}\frac{\langle n|\partial_{k_α}H|m\rangle
\langle m|\partial_{k_β}H|n\rangle}{ε_n-ε_m}.
```
Both Hamiltonian derivatives use ForwardDiff at fixed plane waves and density.
The complete eigenbasis of the finite Hamiltonian block is required, so there is
no hidden empty-band truncation. `carrier=:hole` reverses curvature at a valence
maximum. Tensors are in electron-mass units. Hubbard second derivatives, hybrids,
meta-GGA and degenerate-band k⋅p models are not implemented here.
Pseudopotential masses exactly at Γ also require radial second-derivative origin
limits not provided by the existing AD rules; use nearby nondegenerate points
with a documented k→0 convergence study instead.
"""
function compute_effective_mass(data; band, ik=1, carrier=:electron, degeneracy_tol=1e-8)
    basis = data.basis
    _check_optical_basis(basis)
    carrier in (:electron, :hole) || throw(ArgumentError("Choose electron or hole."))
    any(t -> t isa TermHubbard, basis.terms) && error(
        "Reference effective masses do not yet support Hubbard second derivatives.")
    kpt, ψ, ε = basis.kpoints[ik], data.ψ[ik], data.eigenvalues[ik]
    1 ≤ band ≤ length(ε) ||
        throw(ArgumentError("The effective-mass band is out of range."))
    if any(t -> t isa TermAtomicNonlocal, basis.terms) &&
       any(iszero, Gplusk_vectors_cart(basis, kpt))
        error("Pseudopotential mass at Γ needs radial second-derivative origin limits; " *
              "evaluate nearby k points and converge k→0.")
    end
    size(ψ, 1) == size(ψ, 2) == length(ε) || error(
        "Reference masses require the complete eigenbasis of the finite Hamiltonian block.")
    all(m -> m == band || abs(ε[band]-ε[m]) > degeneracy_tol, eachindex(ε)) ||
        error("Degenerate bands require a subspace k⋅p effective-mass treatment.")
    # Reuse velocity's operator checks without differentiating a full square band matrix.
    probe = merge(data, (; ψ=[ψk[:, 1:1] for ψk in data.ψ],
                         occupation=[f[1:1] for f in data.occupation]))
    compute_velocity(probe)
    ψn = ψ[:, band:band]
    function H(δk, left, right)
        shifted = _shift_kpoint(basis, kpt, δk)
        sum(zip(basis.model.term_types, basis.terms)) do (builder, term)
            _optical_kmatrix(builder, term, basis, shifted, left, right; hubbard_n=nothing)
        end
    end
    δk0 = zeros(eltype(basis), 3)
    curvature = ForwardDiff.hessian(δk -> real(only(H(δk, ψn, ψn))), δk0)
    J = ForwardDiff.jacobian(zeros(eltype(basis), 3)) do δk
        h = H(δk, ψ, ψn)
        vcat(vec(real.(h)), vec(imag.(h)))
    end
    v = J[1:length(ε), :]+im*J[length(ε)+1:end, :]
    for α = 1:3, β = 1:3, m in eachindex(ε)
        m == band && continue
        curvature[α, β] += 2real(conj(v[m, α])*v[m, β])/(ε[band]-ε[m])
    end
    inverse_mass = (carrier == :hole ? -1 : 1)*curvature
    principal = eigen(Symmetric(inverse_mass))
    (; inverse_mass, mass_tensor=inv(inverse_mass),
     principal_masses=inv.(principal.values), directions=principal.vectors, band, ik, carrier)
end
