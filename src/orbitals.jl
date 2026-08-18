using Random  # Used to have a generic API for CPU and GPU computations alike: see random_orbitals

# Bloch-wave representation API. Existing one-component modes keep their Matrix storage,
# while multicomponent states use (component, G, band) arrays. Dense linear algebra uses
# `as_matrix`, which flattens the first two axes without copying.
n_components(ψk::AbstractMatrix) = 1
n_components(ψk::AbstractArray{<:Any,3}) = size(ψk, 1)

"""Return an orbital array with an explicit `(component, G, band)` layout."""
as_components(ψk::AbstractMatrix) = reshape(ψk, 1, size(ψk)...)
as_components(ψk::AbstractArray{<:Any,3}) = ψk

"""Return the zero-copy `(component*G, band)` matrix lens used by eigensolvers."""
as_matrix(ψk::AbstractVector) = reshape(ψk, :, 1)
as_matrix(ψk::AbstractMatrix) = ψk
as_matrix(ψk::AbstractArray{<:Any,3}) = reshape(ψk, :, size(ψk, 3))
_n_bands(ψk) = size(as_matrix(ψk), 2)

"""Return the `(G, band)` view of component `σ` in an orbital array."""
component_view(ψk::AbstractMatrix, σ::Integer) =
    σ == 1 ? ψk : throw(BoundsError(ψk, (σ, :, :)))
component_view(ψk::AbstractArray{<:Any,3}, σ::Integer) = @view ψk[σ, :, :]

_each_component(ψk, ::Val{C}) where {C} = ntuple(σ -> component_view(ψk, σ), Val(C))
"""Return a tuple containing the `(G, band)` view of every orbital component."""
each_component(ψk) = _each_component(ψk, Val(n_components(ψk)))

"""
    spinors(ψk)

Return a zero-copy `(G, band)` array whose entries are static component vectors.
"""
function spinors(ψk)
    ψ_components = as_components(ψk)
    C = size(ψ_components, 1)
    reinterpret(reshape, SVector{C,eltype(ψ_components)}, ψ_components)
end

"""
    spin_projections(ψk)

Return the expectation values `(⟨σx⟩, ⟨σy⟩, ⟨σz⟩)` for every band of a
two-component orbital array. The result has size `(n_bands, 3)`.
"""
function spin_projections(ψk)
    n_components(ψk) == 2 || throw(ArgumentError(
        "Spin projections require two-component orbitals."))
    # Component views are strided, which would miss the GPU-specialised columnwise_dots
    # and fall back to scalar indexing, so materialise them.
    ψup, ψdown = copy.(each_component(ψk))
    normalization = vec(sum(abs2, ψup; dims=1) .+ sum(abs2, ψdown; dims=1))
    cross = columnwise_dots(ψup, ψdown)
    hcat(2real.(cross) ./ normalization,
         2imag.(cross) ./ normalization,
         (vec(sum(abs2, ψup; dims=1)) .- vec(sum(abs2, ψdown; dims=1)))
         ./ normalization)
end
spin_projections(basis, ψ) = [spin_projections(ψk) for ψk in ψ]

_canonical_orbitals(ψk::AbstractMatrix, ::Val{1}, n_G::Integer) = ψk
function _canonical_orbitals(ψk::AbstractMatrix, ::Val{C}, n_G::Integer) where {C}
    @assert size(ψk, 1) == C * n_G
    reshape(ψk, C, n_G, size(ψk, 2))
end

_select_bands(ψk::AbstractMatrix, bands) = @view ψk[:, bands]
_select_bands(ψk::AbstractArray{<:Any,3}, bands) = @view ψk[:, :, bands]

# Returns the occupied orbitals, the occupation array and optionally the eigenvalues without
# virtual states (or states with small occupation level for metals).
# threshold is a parameter to distinguish between states we want to keep and the
# others when using temperature. It is set to 0.0 by default, to treat with insulators.
function select_occupied_orbitals(basis, ψ, occupation; threshold=0.0)
    mask_occ = occupied_empty_masks(occupation, threshold).mask_occ
    selected_ψ   = [_select_bands(ψk, mask_occ[ik]) for (ik, ψk) in enumerate(ψ)]
    selected_occ = [      occk[mask_occ[ik]]  for (ik, occk) in enumerate(occupation)]

    # if we have an insulator, sanity check that the orbitals we kept are the
    # occupied ones
    if iszero(threshold)
        model   = basis.model
        n_spin  = model.n_spin_components
        n_bands = div(model.n_electrons, n_spin * filled_occupation(model), RoundUp)
        @assert n_bands == _n_bands(selected_ψ[1])
    end
    (; ψ=selected_ψ, occupation=selected_occ)
end

# Packing routine to store all bands ψ using same-sized matrices.
# This is useful for serialisation to disk
function blockify_ψ(basis::PlaneWaveBasis, ψ::AbstractVector)
    # Determine largest rectangular grid on which all bands can live
    n_G_vectors = [length(kpt.mapping) for kpt in basis.kpoints]
    max_n_G = mpi_max(maximum(n_G_vectors), basis.comm_kpts)

    # Zero-pad each ψk to the full stored rectangle
    ψ = map(ψ) do ψk
        ψk_components = as_components(ψk)
        C, n_G, n_bands = size(ψk_components)
        ψk_full_components = similar(ψk_components, C, max_n_G, n_bands)
        fill!(ψk_full_components, 0)
        ψk_full_components[:, 1:n_G, :] = ψk_components
        ψk_full = C == 1 ? reshape(ψk_full_components, max_n_G, n_bands) :
                          ψk_full_components
        ψk_full
    end
    (; ψ, n_G_vectors)
end

# Unpacking routine for the above by truncating each array to the sizes
# given in the second argument.
function unblockify_ψ(ψ_padded::AbstractVector, n_G_vectors::AbstractVector)
    map(n_G_vectors, ψ_padded) do n_Gk, ψk_padded
        if ndims(ψk_padded) == 2
            ψk_padded[1:n_Gk, :]
        else
            ψk_padded[:, 1:n_Gk, :]
        end
    end
end


# Packing routines used in direct_minimization and newton algorithms.
# They pack / unpack sets of ψ's (or compatible arrays, such as hamiltonian
# applies and gradients) to make them compatible to be used in algorithms
# from IterativeSolvers.
# Some care is needed here : some operators (for instance K in newton.jl)
# are real-linear but not complex-linear. To overcome this difficulty, instead of
# seeing them as operators from C^N to C^N, we see them as
# operators from R^2N to R^2N. In practice, this is done with the
# reinterpret function from julia.
# /!\ pack_ψ does not share memory while unpack_ψ does

reinterpret_real(x) = reinterpret(real(eltype(x)), x)
reinterpret_complex(x) = reinterpret(Complex{eltype(x)}, x)

function pack_ψ(ψ)
    # TODO as an optimization, do that lazily? See LazyArrays
    reduce(vcat, [vec(ψk) for ψk in ψ])
end

# Returns pointers into the unpacked ψ
# /!\ The resulting array is only valid as long as the original x is still in live memory.
function unsafe_unpack_ψ(x, sizes_ψ)
    lengths = prod.(sizes_ψ)
    ends = cumsum(lengths)
    # We unsafe_wrap the resulting array to avoid a complicated type for ψ.
    map(1:length(sizes_ψ)) do ik
        unsafe_wrap(Array{complex(eltype(x))},
                    pointer(@views x[ends[ik]-lengths[ik]+1:ends[ik]]),
                    sizes_ψ[ik])
    end
end
unpack_ψ(x, sizes_ψ) = deepcopy(unsafe_unpack_ψ(x, sizes_ψ))

function random_orbitals(basis::PlaneWaveBasis{T}, kpt::Kpoint, howmany::Integer) where {T}
    n_G = length(G_vectors(basis, kpt))
    C = n_components(basis.model)
    orbitals = similar(G_vectors(basis), Complex{T}, C, n_G, howmany)
    randn!(TaskLocalRNG(), orbitals)  # use the RNG on the device if we're using a GPU
    orbitals_matrix = ortho_qr(as_matrix(orbitals))
    _canonical_orbitals(orbitals_matrix, Val(C), n_G)
end

function random_orbitals(basis::AbstractBasis, howmany::Integer)
    map(basis.kpoints) do kpt
        random_orbitals(basis, kpt, howmany)
    end
end
