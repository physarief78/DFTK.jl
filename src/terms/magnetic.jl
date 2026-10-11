# This is pretty naive and probably needs to be thought about a bit more, esp. in the periodic context.
# Right now this is sufficient to reproduce uniform fields for isolated systems.

@doc raw"""
Magnetic term ``A⋅(-i∇)``. It is assumed (but not checked) that ``∇⋅A = 0``.
"""
struct Magnetic
    Afunction::Function  # A(x,y,z) returns [Ax,Ay,Az]
                         # both [x,y,z] and [Ax,Ay,Az] are in *Cartesian* coordinates
end
function (M::Magnetic)(basis)
    n_components(basis.model) == 1 || error(
        "Vector-potential magnetic terms are not implemented for multicomponent orbitals.")
    TermMagnetic(basis, M.Afunction)
end

struct MagneticFromValues
    # Apotential[α] is an array of size fft_size for α=1:3
    Apotential::Vector{<:AbstractArray}
end
function (M::MagneticFromValues)(basis::PlaneWaveBasis)
    n_components(basis.model) == 1 || error(
        "Vector-potential magnetic terms are not implemented for multicomponent orbitals.")
    length(M.Apotential) == 3 || throw(DimensionMismatch(
        "A magnetic vector potential needs three components."))
    all(size(A) == basis.fft_size for A in M.Apotential) || throw(DimensionMismatch(
        "Magnetic vector-potential components must match the FFT grid."))
    TermMagnetic(M.Apotential)
end

struct TermMagnetic <: TermLinear
    # Apotential[α] is an array of size fft_size for α=1:3
    Apotential::Vector{<:AbstractArray}
end
function TermMagnetic(basis::PlaneWaveBasis{T}, Afunction::Function) where {T}
    Apotential = [zeros(T, basis.fft_size) for α = 1:3]
    N1, N2, N3 = basis.fft_size
    rvecs = collect(r_vectors_cart(basis))
    for i = 1:N1
        for j = 1:N2
            for k = 1:N3
                Apotential[1][i, j, k],
                Apotential[2][i, j, k],
                Apotential[3][i, j, k] = Afunction(rvecs[i, j, k])
            end
        end
    end
    TermMagnetic(Apotential)
end

function ene_ops(term::TermMagnetic, basis::PlaneWaveBasis{T}, ψ, occupation;
                 kwargs...) where {T}
    ops = [MagneticFieldOperator(basis, kpoint, term.Apotential)
           for (ik, kpoint) in enumerate(basis.kpoints)]
    if isnothing(ψ) || isnothing(occupation)
        return (; E=T(Inf), ops)
    end

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        E += basis.kweights[ik] * sum(occupation[ik] .* vec(real(columnwise_dots(ψ[ik], ops[ik] * ψ[ik]))))
    end
    E = mpi_sum(E, basis.comm_kpts)

    (; E, ops)
end


@doc raw"""
    MagneticField(field)

External Zeeman field with Hamiltonian contribution ``-\mathbf b(r)\cdot\boldsymbol\sigma``.
`field` can be a constant three-vector, a function of Cartesian position, or an array of
size `(basis.fft_size..., 3)`. The energy is
``E_Z = -\int \mathbf b(r)\cdot\mathbf m(r)\,dr``.

Supported spin modes are `:collinear` and `:full`. In collinear mode the field must
point along z: `MagneticField([0, 0, b])` adds potentials ``V_\uparrow=-b`` and
``V_\downarrow=+b``. Transverse components require `spin_polarization=:full`.

The field is an energy in Hartree: for g = 2, ``b = \mu_B B``, so 1 T corresponds
to approximately 2.1272e-6 Ha. Positive b favours positive z magnetization.
Without spin-orbit coupling a uniform field preserves spatial symmetries;
spatially varying fields disable symmetry reduction. Full-spin models use identity
symmetry in either case.
"""
struct MagneticField{F}
    field::F
end

function _magnetic_field_values(basis::PlaneWaveBasis{T},
                                field::AbstractArray{<:Any,4}) where {T}
    size(field) == (basis.fft_size..., 3) || throw(DimensionMismatch(
        "Magnetic-field values must have size (basis.fft_size..., 3)."))
    to_device(basis.architecture, convert_dual.(T, field))
end
function _magnetic_field_values(basis::PlaneWaveBasis{T}, field::Function) where {T}
    values = Array{T}(undef, basis.fft_size..., 3)
    positions = to_cpu(r_vectors_cart(basis))
    for index in CartesianIndices(basis.fft_size)
        value = field(positions[index])
        length(value) == 3 || throw(DimensionMismatch(
            "A magnetic-field function must return three entries."))
        for α = 1:3
            values[index, α] = convert_dual(T, value[α])
        end
    end
    to_device(basis.architecture, values)
end
function _magnetic_field_values(basis::PlaneWaveBasis{T},
                                field::Union{Tuple,AbstractVector}) where {T}
    length(field) == 3 || throw(DimensionMismatch("A uniform magnetic field needs 3 entries."))
    values = zeros(T, basis.fft_size..., 3)
    for α = 1:3
        values[:, :, :, α] .= convert_dual(T, field[α])
    end
    to_device(basis.architecture, values)
end
function _magnetic_field_values(::PlaneWaveBasis, field)
    throw(ArgumentError("Unsupported magnetic-field representation $(typeof(field))."))
end

function (term::MagneticField)(basis::PlaneWaveBasis)
    spin = basis.model.spin_polarization
    spin in (:collinear, :full) || throw(ArgumentError(
        "MagneticField requires spin_polarization=:collinear or :full."))
    field = _magnetic_field_values(basis, term.field)
    if spin == :collinear && !iszero(@view field[:, :, :, 1:2])
        throw(ArgumentError("For spin_polarization=:collinear the magnetic field must " *
                            "point along z; use :full for transverse fields."))
    end
    TermMagneticField(field)
end

struct TermMagneticField{A} <: TermLinear
    field::A
end

function ene_ops(term::TermMagneticField, basis::PlaneWaveBasis{T}, ψ, occupation;
                 ρ=nothing, kwargs...) where {T}
    if basis.model.spin_polarization == :collinear
        bz = @view term.field[:, :, :, 3]
        potentials = (-bz, +bz)
        ops = [RealSpaceMultiplication(basis, kpt, potentials[kpt.spin])
               for kpt in basis.kpoints]
    else
        potential = zeros_like(term.field, T, basis.fft_size..., 4)
        @views potential[:, :, :, 2:4] .= .-term.field
        ops = [MatrixLocalOperator(basis, kpt, potential) for kpt in basis.kpoints]
    end
    if isnothing(ρ)
        return (; E=T(Inf), ops)
    end
    E = -sum(term.field .* magnetization(basis, ρ)) * basis.dvol
    (; E, ops)
end
