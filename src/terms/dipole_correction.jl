@doc raw"""
    DipoleCorrection(; direction=3, cut=0)

Bengtsson correction for a neutral slab, with the sawtooth discontinuity `cut` in
fractional coordinates in the vacuum. The physical charge dipole is
```math
p=\sum_A Z_A s_A-\int s(r)ρ(r)dr,\quad
E_{dip}=2πp^2/Ω,\quad V_{dip}(r)=-4πp\,s(r)/Ω.
```
`ρ` is the positive electron number density and `s` is distance along the reciprocal
lattice normal, wrapped about `cut+1/2`. Keep the cut away from ions and density.
The correction is not a Coulomb truncation and must not be applied a second time
to already isolated electrostatics. See doi:10.1103/PhysRevB.59.12301.
"""
struct DipoleCorrection
    direction::Int
    cut::Real
    function DipoleCorrection(; direction=3, cut=0)
        direction in 1:3 || throw(ArgumentError("direction must be 1, 2 or 3."))
        new(direction, mod(cut, 1))
    end
end
breaks_symmetries(::DipoleCorrection) = true

struct TermDipoleCorrection{T,A} <: TermNonlinear
    direction::Int
    cut::T
    height::T
    normal::Vec3{T}
    coordinate::A
    ionic_dipole::T
end

# Hooks for an explicit neutralizing sheet, defined on the fractional-electron branch.
_countercharge(::Any) = 0
_countercharge_dipole(::Any, basis, direction, cut, height) = zero(eltype(basis))

function (t::DipoleCorrection)(basis::PlaneWaveBasis{T}) where {T}
    model = basis.model
    charge = sum(charge_ionic, model.atoms; init=zero(T))-model.n_electrons
    charge += sum(_countercharge, model.term_types; init=zero(T))
    abs(charge) < 1e-10 ||
        error("DipoleCorrection requires a neutral cell including countercharge.")
    covector = Vec3(model.inv_lattice[t.direction, :])
    height = inv(norm(covector))
    normal = height*covector
    s(r) = height*(mod(r[t.direction]-t.cut, 1)-1/2)
    coordinate = s.(r_vectors(basis))
    ionic_dipole = sum(zip(model.atoms, model.positions); init=zero(T)) do (atom, r)
        charge_ionic(atom)*s(r)
    end
    ionic_dipole += sum(model.term_types; init=zero(T)) do builder
        _countercharge_dipole(builder, basis, t.direction, t.cut, height)
    end
    TermDipoleCorrection(t.direction, T(t.cut), height, normal, coordinate, ionic_dipole)
end

_slab_dipole(term::TermDipoleCorrection, basis, ρ) =
    term.ionic_dipole - sum(term.coordinate .* total_density(ρ))*basis.dvol

function ene_ops(term::TermDipoleCorrection, basis, ψ, occupation; ρ, kwargs...)
    p = _slab_dipole(term, basis, ρ)
    potential = -4π*p/basis.model.unit_cell_volume .* term.coordinate
    ops = [RealSpaceMultiplication(basis, kpt, potential) for kpt in basis.kpoints]
    (; E=2π*p^2/basis.model.unit_cell_volume, ops)
end

function compute_forces(term::TermDipoleCorrection, basis::PlaneWaveBasis, ψ, occupation;
                        ρ, kwargs...)
    field = -4π*_slab_dipole(term, basis, ρ)/basis.model.unit_cell_volume * term.normal
    [basis.model.lattice'*(charge_ionic(atom)*field) for atom in basis.model.atoms]
end

function apply_kernel(term::TermDipoleCorrection, basis::PlaneWaveBasis, δρ;
                      q=zero(Vec3{eltype(basis)}), kwargs...)
    iszero(q) || error("The dipole-correction response is implemented only at q=0.")
    δp = -sum(term.coordinate .* total_density(δρ))*basis.dvol
    δVρ = zero(δρ)
    δVρ .= -4π*δp/basis.model.unit_cell_volume .* term.coordinate
    (; δVρ)
end
function compute_kernel(term::TermDipoleCorrection, basis::PlaneWaveBasis; kwargs...)
    s = repeat(vec(term.coordinate), basis.model.n_spin_components)
    4π*basis.dvol/basis.model.unit_cell_volume * s*s'
end

compute_dynmat(::TermDipoleCorrection,::AbstractBasis,ψ,occupation;kwargs...) = error(
    "DFPT ionic derivatives of DipoleCorrection are not implemented; " *
    "use finite differences of the supported forces.")
