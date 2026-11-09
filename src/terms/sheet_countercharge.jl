@doc raw"""
    SheetCounterCharge(charge; center=0.85, width=1, direction=3)

Periodic Gaussian sheet of physical charge `charge` (in e), uniformly distributed
in the slab plane. `center` is fractional and `width` is a standard deviation in Bohr:
```math
ρ_{cc,G}=\frac{q}{\sqrt Ω}e^{-σ^2G_⊥^2/2}e^{-iG_⊥z_0},\quad G_∥=0.
```
The electron potential is **negative**, `V_cc,G=-4πρ_cc,G/G²`. The added energy
contains electron–sheet, ion–sheet and sheet self-energy. G=0 is omitted, replacing
the pre-existing uniform neutralizing background with the sheet in the same gauge.
The charge must compensate `n_electrons-sum(Zion)`. Put the sheet in empty vacuum
and converge its distance and width; it is an external classical charge, not an electrode.
"""
struct SheetCounterCharge{T}
    charge::T
    center::T
    width::T
    direction::Int
end
function SheetCounterCharge(charge; center=0.85, width=1, direction=3)
    direction in 1:3 || throw(ArgumentError("direction must be 1, 2 or 3."))
    width = austrip(width)
    width > 0 || throw(ArgumentError("The Gaussian sheet width must be positive."))
    isfinite(charge) || throw(ArgumentError("The sheet charge must be finite."))
    T = promote_type(typeof(float(charge)), typeof(float(center)), typeof(float(width)))
    SheetCounterCharge{T}(charge, mod(center, 1), width, direction)
end
breaks_symmetries(::SheetCounterCharge) = true
_countercharge(t::SheetCounterCharge) = t.charge

struct TermSheetCounterCharge{T,A,C} <: TermLinear
    direction::Int
    charge_density::A
    potential_values::A
    modes::Vector{Int}
    charge_potential::Vector{C}
    constant_energy::T
end

function (t::SheetCounterCharge)(basis::PlaneWaveBasis{T}) where {T}
    basis.architecture isa CPU ||
        error("Reference sheet countercharge requires a CPU basis.")
    injected = basis.model.n_electrons-sum(charge_ionic, basis.model.atoms; init=zero(T))
    abs(injected-t.charge) ≤ 1e-10 || error(
        "Sheet charge must equal the number of injected electrons ($injected).")
    count(x -> x isa SheetCounterCharge, basis.model.term_types) == 1 ||
        error("Use exactly one compensating sheet.")
    ρG = map(G_vectors(basis)) do G
        all(α -> α == t.direction || iszero(G[α]), 1:3) || return zero(Complex{T})
        p = basis.model.recip_lattice*G
        t.charge/sqrt(basis.model.unit_cell_volume)*exp(-t.width^2*norm2(p)/2)*
            cis2pi(-G[t.direction]*t.center)
    end
    φG = map(G_vectors(basis), ρG) do G, ρ
        iszero(G) ? zero(ρ) : 4T(π)/norm2(basis.model.recip_lattice*G)*ρ
    end
    modes = Int[]
    coefficients = Complex{T}[]
    for (G, φ) in zip(G_vectors(basis), φG)
        all(α -> α == t.direction || iszero(G[α]), 1:3) || continue
        push!(modes, G[t.direction])
        push!(coefficients, φ/sqrt(basis.model.unit_cell_volume))
    end
    φ(r) = real(sum(c*cis2pi(n*r[t.direction]) for (n, c) in zip(modes, coefficients)))
    Eion = sum(zip(basis.model.atoms, basis.model.positions); init=zero(T)) do (atom, r)
        charge_ionic(atom)*φ(r)
    end
    Eself = real(dot(ρG, φG))/2
    TermSheetCounterCharge(t.direction, irfft(basis, ρG), -irfft(basis, φG),
                           modes, coefficients, Eion+Eself)
end

function ene_ops(term::TermSheetCounterCharge, basis, ψ, occupation; ρ, kwargs...)
    E = sum(total_density(ρ).*term.potential_values)*basis.dvol + term.constant_energy
    ops = [RealSpaceMultiplication(basis, kpt, term.potential_values)
           for kpt in basis.kpoints]
    (; E, ops)
end
function compute_forces(term::TermSheetCounterCharge, basis::PlaneWaveBasis, ψ, occupation;
                        kwargs...)
    map(zip(basis.model.atoms, basis.model.positions)) do (atom, r)
        derivative = real(sum(2π*im*n*c*cis2pi(n*r[term.direction])
                              for (n, c) in zip(term.modes, term.charge_potential)))
        Vec3(ntuple(α -> α == term.direction ? -charge_ionic(atom)*derivative :
                     zero(derivative), 3))
    end
end
_electrostatic_potential(term::TermSheetCounterCharge, basis, ρ) = term.potential_values
function _countercharge_dipole(t::SheetCounterCharge, basis, direction, cut, height)
    t.direction == direction ||
        error("Sheet and dipole correction must use the same normal.")
    term = t(basis)
    s = map(r -> height*(mod(r[direction]-cut, 1)-1/2), r_vectors(basis))
    sum(s.*term.charge_density)*basis.dvol
end
