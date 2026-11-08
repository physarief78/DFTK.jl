@doc raw"""
Planar average of a **scalar 3D grid**, ``\bar f(s)=A^{-1}\int_A f(r)dr``.
Returns `coordinate` (Bohr), `values`, `spacing` and `area`. Apply `total_density`
explicitly before averaging a spin-resolved density.
"""
function planar_average(basis,f::AbstractArray{T,3};direction=3) where {T}
    size(f) == basis.fft_size || throw(DimensionMismatch("The grid must match the basis."))
    direction in 1:3 || throw(ArgumentError("direction must be 1, 2 or 3."))
    dims = Tuple(filter(!=(direction),1:3))
    values = vec(sum(f;dims=dims))/prod(basis.fft_size[collect(dims)])
    height = inv(norm(basis.model.inv_lattice[direction,:]))
    spacing = height/length(values)
    (;coordinate=collect(0:length(values)-1)*spacing,values,spacing,
     area=basis.model.unit_cell_volume/height)
end

"""Electron potential energy from ionic, Hartree and explicit electrostatic correction terms."""
function compute_electrostatic_potential(data::NamedTuple)
    basis = data.basis
    V = zeros(eltype(basis),basis.fft_size)
    for term in basis.terms
        potential = _electrostatic_potential(term,basis,data.ρ)
        isnothing(potential) || (V .+= potential)
    end
    V
end
_electrostatic_potential(term,basis,ρ) = nothing
_electrostatic_potential(term::TermAtomicLocal,basis,ρ) = term.potential_values
_electrostatic_potential(term::TermHartree,basis,ρ) =
    irfft(basis,term.poisson_green_coeffs .* fft(basis,total_density(ρ)))
_electrostatic_potential(term::TermDipoleCorrection,basis,ρ) =
    -4π*_slab_dipole(term,basis,ρ)/basis.model.unit_cell_volume .* term.coordinate

@doc raw"""
Work function ``Φ=V_{vac}-ε_F`` (Ha). The user supplies a vacuum window in Bohr
along `direction`; its potential spread must be smaller than `tolerance` (Ha).
For constrained ensembles supply the chosen chemical potential explicitly.
An insulating midgap εF defines a convention, not a unique ionization energy.
"""
function compute_work_function(data;vacuum_window,direction=3,εF=data.εF,tolerance=1e-3)
    εF isa Real || throw(ArgumentError("Select a scalar chemical potential explicitly."))
    profile = planar_average(data.basis,compute_electrostatic_potential(data);direction)
    indices = findall(z -> vacuum_window[1] ≤ z ≤ vacuum_window[2],profile.coordinate)
    length(indices) ≥ 2 || throw(ArgumentError("The vacuum window needs two grid planes."))
    values = profile.values[indices]
    spread = maximum(values)-minimum(values)
    spread ≤ tolerance || error("Vacuum potential is not flat: spread=$spread Ha.")
    vacuum = sum(values)/length(values)
    (;work_function=vacuum-εF,vacuum,εF,spread,profile,indices)
end

@doc raw"""
Electron redistribution ``Δρ=ρ_{interface}-\sum_Fρ_F``. All fragments must use the
same lattice and FFT grid, and must be evaluated in the frozen interface geometry.
Positive Δρ is electron accumulation. No resampling or charge normalization is done.
"""
function density_difference(data,fragments::AbstractVector)
    Δρ = copy(total_density(data.ρ))
    for fragment in fragments
        fragment.basis.model.lattice ≈ data.basis.model.lattice ||
            throw(ArgumentError("Fragment lattice must match the interface."))
        fragment.basis.fft_size == data.basis.fft_size ||
            throw(DimensionMismatch("Fragment FFT grid must match the interface."))
        Δρ .-= total_density(fragment.ρ)
    end
    (;Δρ,excess_electrons=sum(Δρ)*data.basis.dvol)
end

@doc raw"""
Plane-integrated cumulative electron redistribution ``Q(s)=A\int_0^s\bar{Δρ}(z)dz``.
The rectangle sum uses the same quadrature as DFTK's electron count. It is not a
unique atom or fragment charge partition.
"""
function cumulative_density(basis,Δρ;direction=3)
    profile = planar_average(basis,Δρ;direction)
    (;profile...,electrons=cumsum(profile.values)*(profile.area*profile.spacing))
end

@doc raw"""
Valence Hirshfeld charges ``q_A=Z_A-\int ρ(r) w_A(r)dr``, with
``w_A=ρ_A/\sum_Bρ_B`` from neutral pseudo-atomic valence densities.
Fourier ringing in reference densities is clipped to zero before partitioning.
`method=ValenceDensityPseudo()` fails if a pseudopotential lacks valence densities;
`ValenceDensityAuto()` explicitly permits Gaussian references as an alternative.
This valence partition is basis- and reference-dependent, not an observable.
"""
function compute_hirshfeld(data;method=ValenceDensityPseudo())
    basis = data.basis
    natoms = length(basis.model.atoms)
    natoms > 0 || throw(ArgumentError("Hirshfeld charges require atoms."))
    references = map(1:natoms) do i
        coefficients = zeros(eltype(basis),natoms)
        coefficients[i] = 1
        max.(atomic_density_superposition(basis,method;coefficients),0)
    end
    promolecule = sum(references)
    ρ = total_density(data.ρ)
    any((promolecule .== 0) .& (abs.(ρ) .> 1e-12)) &&
        error("Reference density vanishes where electron density is nonzero.")
    populations = map(references) do reference
        weights = reference ./ ifelse.(promolecule .> 0,promolecule,one(eltype(basis)))
        sum(weights .* ρ)*basis.dvol
    end
    charges = charge_ionic.(basis.model.atoms)-populations
    (;charges,populations,method)
end

"""Sum atomic-orbital PDOS over the union of the selected atom indices."""
function compute_fragment_pdos(εs,data,indices;kwargs...)
    atoms = unique(indices)
    all(i -> 1 ≤ i ≤ length(data.basis.model.atoms),atoms) ||
        throw(ArgumentError("Fragment indices are outside the atom list."))
    result = compute_pdos(εs,data;kwargs...)
    selected = findall(label -> label.iatom in atoms,result.projector_labels)
    (;εs,pdos=dropdims(sum(result.pdos[:,selected,:];dims=2);dims=2),atoms)
end

"""Export a scalar FFT grid to Gaussian cube, with lattice, origin and atoms in Bohr."""
function write_cube(filename,basis,values::AbstractArray{T,3};
                    comment="DFTK scalar field, atomic units") where {T <: Real}
    size(values) == basis.fft_size || throw(DimensionMismatch("Grid does not match basis."))
    occursin('\n',comment) && throw(ArgumentError("The cube comment must be one line."))
    open(filename,"w") do io
        println(io,comment,"\nGenerated by DFTK; Bohr grid")
        @printf(io,"%5d %13.6f %13.6f %13.6f\n",length(basis.model.atoms),0,0,0)
        for α = 1:3
            v = basis.model.lattice[:,α]/basis.fft_size[α]
            @printf(io,"%5d %13.6f %13.6f %13.6f\n",basis.fft_size[α],v...)
        end
        for (atom,r) in zip(basis.model.atoms,basis.model.positions)
            @printf(io,"%5d %13.6f %13.6f %13.6f %13.6f\n",charge_nuclear(atom),
                    charge_ionic(atom),(basis.model.lattice*r)...)
        end
        n = 0
        for i in axes(values,1),j in axes(values,2),k in axes(values,3)
            @printf(io," %13.6e",values[i,j,k])
            n += 1
            n % 6 == 0 && println(io)
        end
        n % 6 == 0 || println(io)
    end
    filename
end
