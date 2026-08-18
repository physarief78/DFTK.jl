### A Hamiltonian is composed of blocks (kpoints), which have a list of RealFourierOperator
# corresponding to each term
# This is the "high-level" interface, provided for convenience

abstract type HamiltonianBlock end

# Generic HamiltonianBlock
struct GenericHamiltonianBlock{Tcomponents} <: HamiltonianBlock
    basis::PlaneWaveBasis
    kpoint::Kpoint

    # The operators are vectors of RealFourierOperator,
    # not typed because of type invariance issues.
    operators::Vector  # the original list of RealFourierOperator
                       # (as many as there are terms), kept for easier exploration
    optimized_operators::Vector  # Optimized list of RealFourierOperator, for application

    component_count::Tcomponents

    scratch  # dummy field
end

"""A more optimized HamiltonianBlock for the important case of a DFT Hamiltonian."""
struct DftHamiltonianBlock{Tbasis<:PlaneWaveBasis,
                           Tkpoint<:Kpoint,
                           Tlocal<:RealSpaceMultiplication,
                           Tmatrixlocal<:Union{Nothing,MatrixLocalOperator},
                           TdivAgrad<:Union{Nothing,DivAgradOperator},
                           Tcomponents,
                           Tscratch} <: HamiltonianBlock
    basis::Tbasis
    kpoint::Tkpoint
    operators::Vector

    # Individual operators for easy access
    fourier_op::FourierMultiplication
    local_op::Tlocal
    matrix_local_op::Tmatrixlocal
    nonlocal_op::Union{Nothing,NonlocalOperator}
    divAgrad_op::TdivAgrad
    component_count::Tcomponents

    scratch::Tscratch  # Pre-allocated scratch arrays for fast application
end

function HamiltonianBlock(basis, kpoint, operators; scratch=nothing)
    optimized_operators = optimize_operators(operators)
    fourier_ops  = filter(o -> o isa FourierMultiplication,   optimized_operators)
    real_ops     = filter(o -> o isa RealSpaceMultiplication, optimized_operators)
    matrix_ops   = filter(o -> o isa MatrixLocalOperator,     optimized_operators)
    nonlocal_ops = filter(o -> o isa NonlocalOperator,        optimized_operators)
    divAgrad_ops = filter(o -> o isa DivAgradOperator,        optimized_operators)

    n_ops_grouped = (length(fourier_ops) + length(real_ops) + length(matrix_ops)
                     + length(nonlocal_ops) + length(divAgrad_ops))
    is_dft_ham = (   length(fourier_ops) == 1 && length(real_ops) == 1
                  && length(matrix_ops) < 2 && length(nonlocal_ops) < 2
                  && length(divAgrad_ops) < 2
                  && n_ops_grouped == length(optimized_operators))
    component_count = Val(n_components(basis.model))
    if is_dft_ham
        scratch = @something scratch _ham_allocate_scratch(basis)
        matrix_local_op = isempty(matrix_ops) ? nothing : only(matrix_ops)
        nonlocal_op = isempty(nonlocal_ops) ? nothing : only(nonlocal_ops)
        divAgrad_op = isempty(divAgrad_ops) ? nothing : only(divAgrad_ops)
        DftHamiltonianBlock(basis, kpoint, operators,
                            only(fourier_ops), only(real_ops),
                            matrix_local_op, nonlocal_op, divAgrad_op,
                            component_count, scratch)
    else
        GenericHamiltonianBlock(basis, kpoint, operators, optimized_operators,
                                component_count, nothing)
    end
end
function _ham_allocate_scratch(basis::PlaneWaveBasis{T}) where {T}
    _ham_allocate_scratch(basis, Val(n_components(basis.model)))
end
function _ham_allocate_scratch(basis::PlaneWaveBasis{T}, ::Val{1}) where {T}
    map(1:Threads.nthreads()) do _
        ψ_reals = (zeros_like(G_vectors(basis), complex(T), basis.fft_size...),)
        (; ψ_reals, Hψ_reals=nothing)
    end
end
function _ham_allocate_scratch(basis::PlaneWaveBasis{T}, ::Val{C}) where {T,C}
    map(1:Threads.nthreads()) do _
        ψ_reals = ntuple(Val(C)) do _
            zeros_like(G_vectors(basis), complex(T), basis.fft_size...)
        end
        Hψ_reals = ntuple(Val(C)) do _
            zeros_like(G_vectors(basis), complex(T), basis.fft_size...)
        end
        (; ψ_reals, Hψ_reals)
    end
end

Base.:*(H::HamiltonianBlock, ψ) = mul!(similar(ψ), H, ψ)
Base.eltype(block::HamiltonianBlock) = complex(eltype(block.basis))
Base.size(block::HamiltonianBlock, i::Integer) = i < 3 ? size(block)[i] : 1
function Base.size(block::HamiltonianBlock)
    n_G = length(G_vectors(block.basis, block.kpoint))
    C = n_components(block.basis.model)
    (C * n_G, C * n_G)
end
function random_orbitals(hamk::HamiltonianBlock, howmany::Integer)
    as_matrix(random_orbitals(hamk.basis, hamk.kpoint, howmany))
end

Base.Array(block::HamiltonianBlock)  = Matrix(block)
_operator_matrix(op::MatrixLocalOperator, component_count::Val) = Matrix(op)
_operator_matrix(op, ::Val{1}) = Matrix(op)
function _operator_matrix(op, ::Val{C}) where {C}
    kron(Matrix(op), Matrix{eltype(op.basis)}(I, C, C))
end
Base.Matrix(block::HamiltonianBlock) =
    sum(op -> _operator_matrix(op, block.component_count), block.operators)
Base.Matrix(block::GenericHamiltonianBlock) =
    sum(op -> _operator_matrix(op, block.component_count), block.optimized_operators)

"""Represents a matrix-free Hamiltonian discretized in a given plane-wave basis."""
struct Hamiltonian
    basis::PlaneWaveBasis
    blocks::Vector{HamiltonianBlock}
end

Base.getindex(ham::Hamiltonian, index) = ham.blocks[index]

function LinearAlgebra.mul!(Hψ, H::Hamiltonian, ψ)
    for ik = 1:length(H.basis.kpoints)
        mul!(Hψ[ik], H.blocks[ik], ψ[ik])
    end
    Hψ
end
function Base.:*(H::Hamiltonian, ψ)
    # This allocates new memory for the result of promoted eltype
    result = one(eltype(H.basis)) * ψ
    mul!(result, H, ψ)
end

# Loop through bands, IFFT to get ψ in real space, loop through terms, FFT and accumulate into Hψ
# For the common DftHamiltonianBlock there is an optimized version below
@timing "Hamiltonian multiplication" function LinearAlgebra.mul!(Hψ::AbstractArray,
                                                                 H::GenericHamiltonianBlock,
                                                                 ψ::AbstractArray)
    # Barrier on the component count, such that the per-band storage tuples built below
    # have a length known at compile time.
    _mul_generic_hamiltonian!(Hψ, H, ψ, H.component_count)
end

# Apply one operator to all components of a band. Operators that couple the components
# overload this on their own type, so that the loop below needs no branching on operators.
function _apply_operator_components!(storage, op, ψ_fouriers, ::Val{C}) where {C}
    for σ = 1:C
        apply!((; fourier=storage.Hψ_fouriers[σ], real=storage.Hψ_reals[σ]), op,
               (; fourier=ψ_fouriers[σ], real=storage.ψ_reals[σ]))
    end
end
function _apply_operator_components!(storage, op::MatrixLocalOperator, ψ_fouriers, ::Val)
    apply!((; fourier=nothing, real=storage.Hψ_reals), op,
           (; fourier=nothing, real=storage.ψ_reals))
end

function _mul_generic_hamiltonian!(Hψ::AbstractArray, H::GenericHamiltonianBlock,
                                   ψ::AbstractArray, ::Val{C}) where {C}
    ψ_matrix = as_matrix(ψ)
    Hψ_matrix = as_matrix(Hψ)
    n_G = length(G_vectors(H.basis, H.kpoint))
    size(ψ_matrix, 1) == C * n_G || throw(DimensionMismatch(
        "Orbital and Hamiltonian dimensions do not agree."))
    ψ_components = reshape(ψ_matrix, C, n_G, size(ψ_matrix, 2))
    Hψ_components = reshape(Hψ_matrix, C, n_G, size(Hψ_matrix, 2))

    function allocate_local_storage()
        T = eltype(H.basis)
        Hψ_fouriers = ntuple(Val(C)) do _
            similar(Hψ_matrix, n_G)
        end
        ψ_reals = ntuple(Val(C)) do _
            similar(ψ_matrix, complex(T), H.basis.fft_size...)
        end
        Hψ_reals = ntuple(Val(C)) do _
            similar(Hψ_matrix, complex(T), H.basis.fft_size...)
        end
        (; Hψ_fouriers, ψ_reals, Hψ_reals,
           to = TimerOutput())
    end
    storages = parallel_loop_over_range(1:size(ψ_matrix, 2); allocate_local_storage) do iband, storage
        ψ_fouriers = ntuple(σ -> (@view ψ_components[σ, :, iband]), Val(C))
        for σ = 1:C
            fill!(storage.Hψ_reals[σ], 0)
            fill!(storage.Hψ_fouriers[σ], 0)
            ifft!(storage.ψ_reals[σ], H.basis, H.kpoint, ψ_fouriers[σ])
        end
        for op in H.optimized_operators
            @timeit storage.to "$(nameof(typeof(op)))" begin
                _apply_operator_components!(storage, op, ψ_fouriers, Val(C))
            end
        end
        for σ = 1:C
            Hψσ = @view Hψ_components[σ, :, iband]
            Hψσ .= storage.Hψ_fouriers[σ]
            fft!(storage.Hψ_fouriers[σ], H.basis, H.kpoint, storage.Hψ_reals[σ])
            Hψσ .+= storage.Hψ_fouriers[σ]
        end
    end
    merge!(DFTK.timer, first(storages).to; tree_point=[t.name for t in DFTK.timer.timer_stack])

    Hψ
end

# Apply the scalar local potential in-place when no component-coupling potential is present.
function _apply_local_operators!(scratch, potential, ::Nothing, scaling)
    for ψ_real in scratch.ψ_reals
        ψ_real .*= potential
    end
    scratch.ψ_reals
end
function _apply_local_operators!(scratch, potential, matrix_op::MatrixLocalOperator,
                                 scaling)
    for (Hψ_real, ψ_real) in zip(scratch.Hψ_reals, scratch.ψ_reals)
        Hψ_real .= potential .* ψ_real
    end
    apply_matrix_local!(scratch.Hψ_reals, matrix_op.potential, scratch.ψ_reals;
                        scaling)
end

function _apply_nonlocal!(Hψ, op::NonlocalOperator, ψ, ::Val{1})
    apply!((; fourier=Hψ, real=nothing), op, (; fourier=ψ, real=nothing))
end
function _apply_nonlocal!(Hψ, op::NonlocalOperator, ψ, ::Val{C}) where {C}
    n_G = length(G_vectors(op.basis, op.kpoint))
    ψ_components = reshape(ψ, C, n_G, size(ψ, 2))
    Hψ_components = reshape(Hψ, C, n_G, size(Hψ, 2))
    for σ = 1:C
        # Projector GEMMs require a contiguous G×band operand. The component view is
        # strided because components are adjacent in memory, so copy into scratch here.
        ψσ = copy(@view ψ_components[σ, :, :])
        Hψσ = zero(ψσ)
        apply!((; fourier=Hψσ, real=nothing), op, (; fourier=ψσ, real=nothing))
        @views Hψ_components[σ, :, :] .+= Hψσ
    end
    Hψ
end

# Preserve the legacy scalar-spin hot path exactly. In particular, avoid constructing
# component-shaped orbital views when the block has neither component coupling nor a
# component axis. The vector lens keeps single-orbital `H * ψ` compatible as well.
function _mul_dft_hamiltonian!(Hψ::AbstractArray, H::DftHamiltonianBlock,
                               ψ::AbstractArray, ::Val{1}, ::Nothing)
    ψ_matrix = as_matrix(ψ)
    Hψ_matrix = as_matrix(Hψ)
    n_G = length(G_vectors(H.basis, H.kpoint))
    size(ψ_matrix, 1) == n_G || throw(DimensionMismatch(
        "Orbital and Hamiltonian dimensions do not agree."))

    n_bands = size(ψ_matrix, 2)
    iszero(n_bands) && return Hψ
    have_divAgrad = !isnothing(H.divAgrad_op)
    if have_divAgrad
        G_plus_k = [map(p -> p[α], Gplusk_vectors_cart(H.basis, H.kpoint)) for α = 1:3]
    end

    local_scaling = (H.basis.fft_grid.fft_normalization
                     * H.basis.fft_grid.ifft_normalization)
    potential = H.local_op.potential .* local_scaling
    storages = map(enumerate(H.scratch)) do (i, scratch)
        (scratch, i == 1 ? DFTK.timer : TimerOutput())
    end

    parallel_loop_over_range(1:n_bands, storages) do iband, (scratch, to)
        ψ_real = only(scratch.ψ_reals)
        @timeit to "local" begin
            ifft!(ψ_real, H.basis, H.kpoint, @view(ψ_matrix[:, iband]); normalize=false)
            ψ_real .*= potential
            fft!(@view(Hψ_matrix[:, iband]), H.basis, H.kpoint, ψ_real; normalize=false)
        end

        if have_divAgrad
            @timeit to "divAgrad" begin
                apply!((; fourier=@view(Hψ_matrix[:, iband]), real=nothing),
                       H.divAgrad_op,
                       (; fourier=@view(ψ_matrix[:, iband]), real=nothing);
                       ψ_real, G_plus_k)
            end
        end
    end

    Hψ_matrix .+= H.fourier_op.multiplier .* ψ_matrix
    if !isnothing(H.nonlocal_op)
        @timing "nonlocal" begin
            _apply_nonlocal!(Hψ_matrix, H.nonlocal_op, ψ_matrix, H.component_count)
        end
    end
    Hψ
end

# Fast version, specialized on DFT models. Minimizes the number of FFTs and allocations
function _mul_dft_hamiltonian!(Hψ::AbstractArray, H::DftHamiltonianBlock,
                               ψ::AbstractArray, ::Val{C}, _matrix_local_op) where {C}
    ψ_matrix = as_matrix(ψ)
    Hψ_matrix = as_matrix(Hψ)
    n_G = length(G_vectors(H.basis, H.kpoint))
    size(ψ_matrix, 1) == C * n_G || throw(DimensionMismatch(
        "Orbital and Hamiltonian dimensions do not agree."))
    ψ_components = reshape(ψ_matrix, C, n_G, size(ψ_matrix, 2))
    Hψ_components = reshape(Hψ_matrix, C, n_G, size(Hψ_matrix, 2))

    n_bands = size(ψ_matrix, 2)
    iszero(n_bands) && return Hψ  # Nothing to do if ψ empty
    have_divAgrad = !isnothing(H.divAgrad_op)
    if have_divAgrad
        # TODO: It is very beneficial to precompute G_plus_k here, rather than for each band.
        #       Extra performance could probably be gained by storing this in the HamiltonianBlock
        #       as a scratch array. Is it worth the complication and extra memory use?
        # Precompute G_plus_k for DivAgradOperator
        G_plus_k = [map(p -> p[α], Gplusk_vectors_cart(H.basis, H.kpoint)) for α = 1:3]
    end

    # Notice that we use unnormalized plans for extra speed
    local_scaling = (H.basis.fft_grid.fft_normalization
                     * H.basis.fft_grid.ifft_normalization)
    potential = H.local_op.potential .* local_scaling

    # Give main timer to first thread and a dummy to the others
    storages = map(enumerate(H.scratch)) do (i, scratch)
        (scratch, i == 1 ? DFTK.timer : TimerOutput())
    end

    parallel_loop_over_range(1:n_bands, storages) do iband, (scratch, to)
        @timeit to "local" begin
            for σ = 1:C
                ifft!(scratch.ψ_reals[σ], H.basis, H.kpoint,
                      @view(ψ_components[σ, :, iband]); normalize=false)
            end
            output_reals = _apply_local_operators!(scratch, potential,
                                                   H.matrix_local_op, local_scaling)
            for σ = 1:C
                fft!(@view(Hψ_components[σ, :, iband]), H.basis, H.kpoint,
                     output_reals[σ]; normalize=false)
            end
        end

        if have_divAgrad
            @timeit to "divAgrad" begin
                for σ = 1:C
                    apply!((; fourier=@view(Hψ_components[σ, :, iband]), real=nothing),
                           H.divAgrad_op,
                           (; fourier=@view(ψ_components[σ, :, iband]), real=nothing);
                           ψ_real=scratch.ψ_reals[σ], G_plus_k)
                end
            end
        end
    end

    # Kinetic term
    for σ = 1:C
        @views Hψ_components[σ, :, :] .+= H.fourier_op.multiplier .* ψ_components[σ, :, :]
    end

    # Apply the nonlocal operator.
    if !isnothing(H.nonlocal_op)
        @timing "nonlocal" begin
            _apply_nonlocal!(Hψ_matrix, H.nonlocal_op, ψ_matrix, H.component_count)
        end
    end

    Hψ
end

@timing "DftHamiltonian multiplication" function LinearAlgebra.mul!(Hψ::AbstractArray,
                                                                    H::DftHamiltonianBlock,
                                                                    ψ::AbstractArray)
    _mul_dft_hamiltonian!(Hψ, H, ψ, H.component_count, H.matrix_local_op)
end


"""
Get energies and Hamiltonian
kwargs is additional info that might be useful for the energy terms to precompute
(eg the density ρ)
"""
@timing function energy_hamiltonian(basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    @assert isnothing(ψ) || length(ψ) == length(basis.kpoints)

    # it: index into terms, ik: index into kpoints
    @timing "ene_ops" ene_ops_arr = [ene_ops(term, basis, ψ, occupation; kwargs...)
                                     for term in basis.terms]
    term_names = [string(nameof(typeof(term))) for term in basis.model.term_types]
    energy_values  = [eh.E for eh in ene_ops_arr]
    operators = [eh.ops for eh in ene_ops_arr]         # operators[it][ik]

    # flatten the inner arrays in case a term returns more than one operator
    function flatten(arr)
        ret = []
        for a in arr
            if a isa RealFourierOperator
                push!(ret, a)
            else
                push!(ret, a...)
            end
        end
        ret
    end
    scratch = _ham_allocate_scratch(basis)
    hks_per_k = [flatten([blocks[ik] for blocks in operators])
                 for ik = 1:length(basis.kpoints)]      # hks_per_k[ik][it]
    ham = Hamiltonian(basis, [HamiltonianBlock(basis, kpt, hks; scratch)
                              for (hks, kpt) in zip(hks_per_k, basis.kpoints)])
    energies = Energies(term_names, energy_values)
    (; energies, ham)
end

"""
Faster version than energy_hamiltonian for cases where only the energy is needed.
"""
@timing function energy(basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    energy_values = [energy(term, basis, ψ, occupation; kwargs...) for term in basis.terms]
    term_names = [string(nameof(typeof(term))) for term in basis.model.term_types]
    (; energies=Energies(term_names, energy_values))
end

function Hamiltonian(basis::PlaneWaveBasis; ψ=nothing, occupation=nothing, kwargs...)
    energy_hamiltonian(basis, ψ, occupation; kwargs...).ham
end

"""
Get the total local potential of the given Hamiltonian, in real space
in the spin components.
"""
function total_local_potential(ham::Hamiltonian)
    C = n_components(ham.basis.model)
    if C > 1
        # Full-spin models have a single k-point channel and C² potential fields.
        return total_local_potential(first(ham.blocks))
    end

    pots = map(1:ham.basis.model.n_spin_components) do σ
        # Get the potential from the first Hamiltonian block of this spin component
        # (works since all local potentials are the same)
        i_σ = first(krange_spin(ham.basis, σ))
        total_local_potential(ham.blocks[i_σ])
    end
    cat(pots..., dims=4)
end

_operators_for_local_potential(Hk::DftHamiltonianBlock) = Hk.operators
_operators_for_local_potential(Hk::GenericHamiltonianBlock) = Hk.optimized_operators

function _total_local_potential(operators, basis::PlaneWaveBasis, ::Val{1})
    scalar_potentials = [op.potential for op in operators
                         if op isa RealSpaceMultiplication]
    isempty(scalar_potentials) && error("Hamiltonian has no local potential.")
    reduce(+, scalar_potentials)
end
function _total_local_potential(operators, basis::PlaneWaveBasis{T}, ::Val{C}) where {T,C}
    potential = zeros_like(G_vectors(basis), T, basis.fft_size..., C^2)
    for op in operators
        if op isa RealSpaceMultiplication
            @views potential[:, :, :, 1] .+= op.potential
        elseif op isa MatrixLocalOperator
            potential .+= op.potential
        end
    end
    potential
end
function total_local_potential(Hk::HamiltonianBlock)
    _total_local_potential(_operators_for_local_potential(Hk), Hk.basis,
                           Hk.component_count)
end

"""
Returns a new Hamiltonian with local potential replaced by the given one
"""
function hamiltonian_with_total_potential(ham::Hamiltonian, V)
    @assert size(V, 4) == n_density_components(ham.basis.model)
    C = n_components(ham.basis.model)
    newblocks = if C == 1
        [hamiltonian_with_total_potential(Hk, V[:, :, :, Hk.kpoint.spin])
         for Hk in ham.blocks]
    else
        [hamiltonian_with_total_potential(Hk, V) for Hk in ham.blocks]
    end
    Hamiltonian(ham.basis, newblocks)
end
function hamiltonian_with_total_potential(Hk::HamiltonianBlock, V)
    operators = [op for op in Hk.operators
                 if !(op isa RealSpaceMultiplication || op isa MatrixLocalOperator)]
    if Hk.component_count isa Val{1}
        push!(operators, RealSpaceMultiplication(Hk.basis, Hk.kpoint, V))
    else
        scalar = @view V[:, :, :, 1]
        matrix = copy(V)
        @views matrix[:, :, :, 1] .= 0
        push!(operators, RealSpaceMultiplication(Hk.basis, Hk.kpoint, scalar))
        push!(operators, MatrixLocalOperator(Hk.basis, Hk.kpoint, matrix))
    end
    HamiltonianBlock(Hk.basis, Hk.kpoint, operators; Hk.scratch)
end
