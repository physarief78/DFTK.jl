### Linear operators operating on quantities in real or fourier space
# This is the optimized low-level interface. Use the functions in Hamiltonian for high-level usage

"""
Linear operators that act on tuples (real, fourier)
The main entry point is `apply!(out, op, in)` which performs the operation `out += op*in`
where `out` and `in` are named tuples `(; real, fourier)`.
They also implement `mul!` and `Matrix(op)` for exploratory use.
"""
abstract type RealFourierOperator end
# RealFourierOperator contain fields `basis` and `kpoint`

Base.Array(op::RealFourierOperator) = Matrix(op)

# Slow fallback for getting operator as a dense matrix
function Base.Matrix(op::RealFourierOperator)
    T = complex(eltype(op.basis))
    n_G = length(G_vectors(op.basis, op.kpoint))
    H = zeros(T, n_G, n_G)
    ψ = zeros(T, n_G)
    @views for i in 1:n_G
        ψ[i] = one(T)
        mul!(H[:, i], op, ψ)
        ψ[i] = zero(T)
    end
    H
end

# Unoptimized fallback, intended for exploratory use only.
# For performance, call through Hamiltonian which saves FFTs.
function LinearAlgebra.mul!(Hψ::AbstractVector, op::RealFourierOperator, ψ::AbstractVector)
    ψ_real = ifft(op.basis, op.kpoint, ψ)
    Hψ_fourier = similar(ψ)
    Hψ_real = similar(ψ_real)
    Hψ_fourier .= 0
    Hψ_real .= 0
    apply!((; real=Hψ_real, fourier=Hψ_fourier),
           op,
           (; real=ψ_real, fourier=ψ))
    Hψ .= Hψ_fourier .+ fft(op.basis, op.kpoint, Hψ_real)
    Hψ
end
function LinearAlgebra.mul!(Hψ::AbstractMatrix, op::RealFourierOperator, ψ::AbstractMatrix)
    @views for i = 1:size(ψ, 2)
        mul!(Hψ[:, i], op, ψ[:, i])
    end
    Hψ
end
Base.:*(op::RealFourierOperator, ψ) = mul!(similar(ψ), op, ψ)

"""
Noop operation: don't do anything.
Useful for energy terms that don't depend on the orbitals at all (eg nuclei-nuclei interaction).
"""
struct NoopOperator{T <: Real} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
end
apply!(Hψ, op::NoopOperator, ψ) = nothing
function Base.Matrix(op::NoopOperator)
    n_Gk = length(G_vectors(op.basis, op.kpoint))
    zeros_like(G_vectors(op.basis), eltype(op.basis), n_Gk, n_Gk)
end

"""
Real space multiplication by a potential:
```math
(Hψ)(r) = V(r) ψ(r).
```
"""
struct RealSpaceMultiplication{T <: Real, AT <: AbstractArray} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    potential::AT
end

"""
Pointwise multiplication by a Hermitian matrix-valued potential. The last dimension of
`potential` stores generalized Pauli fields, so it has `C²` entries for `C` wavefunction
components.
"""
struct MatrixLocalOperator{T <: Real,AT <: AbstractArray,C} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    potential::AT
    component_count::Val{C}
end
function MatrixLocalOperator(basis::PlaneWaveBasis{T}, kpoint::Kpoint,
                             potential::AbstractArray) where {T}
    C = n_components(basis.model)
    size(potential) == (basis.fft_size..., C^2) || throw(DimensionMismatch(
        "A $C-component matrix potential needs $(C^2) fields on the FFT grid."))
    MatrixLocalOperator{T,typeof(potential),C}(basis, kpoint, potential, Val(C))
end

@views function apply_matrix_local!(Hψ_reals::NTuple{2}, potential,
                                    ψ_reals::NTuple{2}; scaling=1)
    Hψ1, Hψ2 = Hψ_reals
    ψ1, ψ2 = ψ_reals
    v0 = potential[:, :, :, 1]
    vx = potential[:, :, :, 2]
    vy = potential[:, :, :, 3]
    vz = potential[:, :, :, 4]
    Hψ1 .+= scaling .* ((v0 .+ vz) .* ψ1 .+ (vx .- im .* vy) .* ψ2)
    Hψ2 .+= scaling .* ((vx .+ im .* vy) .* ψ1 .+ (v0 .- vz) .* ψ2)
    Hψ_reals
end

@views function apply_matrix_local!(Hψ_reals::NTuple{C}, potential,
                                    ψ_reals::NTuple{C}; scaling=1) where {C}
    pauli = generalized_pauli_matrices(Val(C), real(eltype(potential)))
    for σ = 1:C, τ = 1:C
        matrix_element = sum(1:C^2) do i
            pauli[i][σ, τ] .* potential[:, :, :, i]
        end
        Hψ_reals[σ] .+= scaling .* matrix_element .* ψ_reals[τ]
    end
    Hψ_reals
end

function apply!(Hψ, op::MatrixLocalOperator, ψ)
    apply_matrix_local!(Hψ.real, op.potential, ψ.real)
end

function Base.Matrix(op::MatrixLocalOperator{T,AT,C}) where {T,AT,C}
    pauli = generalized_pauli_matrices(Val(C), T)
    n_G = length(G_vectors(op.basis, op.kpoint))
    matrix = zeros(Complex{T}, C * n_G, C * n_G)
    for i = 1:C^2
        scalar_op = RealSpaceMultiplication(op.basis, op.kpoint,
                                            @view(op.potential[:, :, :, i]))
        matrix .+= kron(Matrix(scalar_op), Matrix(pauli[i]))
    end
    matrix
end

function LinearAlgebra.mul!(Hψ::AbstractVector, op::MatrixLocalOperator{T,AT,C},
                            ψ::AbstractVector) where {T,AT,C}
    n_G = length(G_vectors(op.basis, op.kpoint))
    length(ψ) == C * n_G || throw(DimensionMismatch("Orbital size does not match operator."))
    ψ_components = reshape(ψ, C, n_G)
    Hψ_components = reshape(Hψ, C, n_G)
    ψ_reals = ntuple(Val(C)) do _
        similar(ψ, complex(T), op.basis.fft_size...)
    end
    Hψ_reals = ntuple(Val(C)) do _
        similar(Hψ, complex(T), op.basis.fft_size...)
    end
    for σ = 1:C
        ifft!(ψ_reals[σ], op.basis, op.kpoint, @view(ψ_components[σ, :]))
        fill!(Hψ_reals[σ], 0)
    end
    apply_matrix_local!(Hψ_reals, op.potential, ψ_reals)
    for σ = 1:C
        fft!(@view(Hψ_components[σ, :]), op.basis, op.kpoint, Hψ_reals[σ])
    end
    Hψ
end
function LinearAlgebra.mul!(Hψ::AbstractMatrix, op::MatrixLocalOperator,
                            ψ::AbstractMatrix)
    @views for n = 1:size(ψ, 2)
        mul!(Hψ[:, n], op, ψ[:, n])
    end
    Hψ
end
function apply!(Hψ, op::RealSpaceMultiplication, ψ)
    Hψ.real .+= op.potential .* ψ.real
end
function Base.Matrix(op::RealSpaceMultiplication)
    # V(G, G') = <eG|V|eG'> = 1/sqrt(Ω) <e_{G-G'}|V>
    pot_fourier = fft(op.basis, op.potential)
    n_G = length(G_vectors(op.basis, op.kpoint))
    H = zeros(complex(eltype(op.basis)), n_G, n_G)
    for (j, G′) in enumerate(G_vectors(op.basis, op.kpoint))
        for (i, G) in enumerate(G_vectors(op.basis, op.kpoint))
            # G_vectors(basis)[ind_ΔG] = G - G'
            ind_ΔG = index_G_vectors(op.basis, G - G′)
            if isnothing(ind_ΔG)
                error("For full matrix construction, the FFT size must be " *
                      "large enough so that Hamiltonian applications are exact")
            end
            H[i, j] = pot_fourier[ind_ΔG] / sqrt(op.basis.model.unit_cell_volume)
        end
    end
    H
end

@doc raw"""
Fourier space multiplication, like a kinetic energy term:
```math
(Hψ)(G) = {\rm multiplier}(G) ψ(G).
```
"""
struct FourierMultiplication{T <: Real, AT <: AbstractArray} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    multiplier::AT
end
function apply!(Hψ, op::FourierMultiplication, ψ)
    Hψ.fourier .+= op.multiplier .* ψ.fourier
end
Base.Matrix(op::FourierMultiplication) = Array(Diagonal(op.multiplier))

"""
Nonlocal operator in Fourier space in Kleinman-Bylander format,
defined by its projectors P matrix and coupling terms D:
Hψ = PDP' ψ.
"""
struct NonlocalOperator{T <: Real, PT, DT} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    # not typed, can be anything that supports PDP'ψ
    P::PT
    D::DT
end
function apply!(Hψ, op::NonlocalOperator, ψ)
    mul!(Hψ.fourier, op.P, (op.D * (op.P' * ψ.fourier)), 1, 1)
end
Base.Matrix(op::NonlocalOperator) = op.P * (op.D * op.P')

"""
Magnetic field operator A⋅(-i∇).
"""
struct MagneticFieldOperator{T <: Real, AT} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    Apot::AT  # Apot[α][i,j,k] is the A field in (Cartesian) direction α
end
function apply!(Hψ, op::MagneticFieldOperator, ψ)
    # TODO this could probably be better optimized
    for α = 1:3
        iszero(op.Apot[α]) && continue
        pα = [p[α] for p in Gplusk_vectors_cart(op.basis, op.kpoint)]
        ∂αψ_fourier = pα .* ψ.fourier
        ∂αψ_real = ifft(op.basis, op.kpoint, ∂αψ_fourier)
        Hψ.real .+= op.Apot[α] .* ∂αψ_real
    end
end
# TODO Implement  Base.Matrix(op::MagneticFieldOperator)

@doc raw"""
Nonlocal "divAgrad" operator ``-½ ∇ ⋅ (A ∇)`` where ``A`` is a scalar field on the
real-space grid. The ``-½`` is included, such that this operator is a generalisation of the
kinetic energy operator (which is obtained for ``A=1``).
"""
struct DivAgradOperator{T <: Real, AT} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    A::AT
end
function apply!(Hψ, op::DivAgradOperator, ψ;
                ψ_real=zeros_like(ψ.fourier, complex(eltype(op.basis)), op.basis.fft_size...),
                G_plus_k=nothing)
    # ψ_real is a pre-allocated buffer to hold temporary real-space representations of ψ,
    # this function overwrites it.
    if isnothing(G_plus_k)
        # Note: it is wasteful to recompute G_plus_k on every call, ideally passed from outside
        G_plus_k = [map(p -> p[α], Gplusk_vectors_cart(op.basis, op.kpoint)) for α = 1:3]
    end

    # use unnormalized FFT plans for speed
    norm = op.basis.fft_grid.fft_normalization * op.basis.fft_grid.ifft_normalization
    ψ_recip = similar(ψ.fourier, complex(eltype(op.basis)))   # pre-allocate large array
    for α = 1:3
        ψ_recip .= im .* G_plus_k[α] .* ψ.fourier .* norm   # ∂αψ
        ifft!(ψ_real, op.basis, op.kpoint, ψ_recip; normalize=false)
        ψ_real .*= op.A   # A∇ψ
        fft!(ψ_recip, op.basis, op.kpoint, ψ_real; normalize=false)
        Hψ.fourier .-= im .* G_plus_k[α] .* ψ_recip ./ 2
    end
end
# TODO Implement  Base.Matrix(op::DivAgradOperator)

struct ExchangeOperator{T <: Real,Tkernel,Tq,Tpsi,Tocc <: Vector{<:Vector}} <: RealFourierOperator
    basis::PlaneWaveBasis{T}
    kpoint::Kpoint{T}
    interaction_kernels::Tkernel  # Vector{Array{T,3}}: kernel on the FFT cube, one per q
    q_points::Tq                  # Vector{Vec3{T}}: momentum transfers (reduced coords)
    ψ_occ_real::Tpsi              # Store precomputed real-space orbitals
    occ_occ::Tocc                 # occupied occupations (cannot be on the GPU)
end
function apply!(Hψ, op::ExchangeOperator, ψ)
    basis = op.basis
    kpt = op.kpoint
    # kernel_q holds the Fourier components of the interaction kernel at G+q
    for (q, kernel_q) in zip(op.q_points, op.interaction_kernels)
        # k - q is equivalent to the basis k-point k' = k - q + ΔG
        ikp, ΔG = find_equivalent_kpt(basis, kpt.coordinate - q, kpt.spin)
        phase_forward = map(r -> cis2pi(-dot(ΔG, r)), r_vectors(basis))
        phase_reverse = map(r -> cis2pi( dot(ΔG, r)), r_vectors(basis))

        # get occupied orbitals at k' in real-space
        ψkp_real = op.ψ_occ_real[ikp]

        # Hψ = - ∑_n f_n ψ_n(r) ∫ (ψ_n)†(r') * ψ(r') / |r-r'| dr'
        for (n, ψnkp_real) in enumerate(eachslice(ψkp_real, dims=4))
            x_real   = conj(ψnkp_real) .* ψ.real .* phase_forward
            # TODO Some symmetrisation of x_real might be needed here ...
    
            # Compute integral by Poisson solve
            x_four  = fft(basis, x_real) 
            Vx_four = x_four .* kernel_q
            Vx_real = ifft(basis, Vx_four) 
    
            # Exact exchange is quadratic in occupations but linear in spin,
            # hence we need to undo the fact that in DFTK for non-spin-polarized calcuations
            # orbitals are considered as spin orbitals and thus occupations run from 0 to 2
            # We do this by dividing by the filled_occupation.
            fac_nk = op.occ_occ[ikp][n] / filled_occupation(basis.model)
         
            fac_nk *= basis.kweights[ikp] # use k'-weight
    
            # Real-space multiply and accumulate
            Hψ.real .-= fac_nk .* ψnkp_real .* Vx_real .* phase_reverse
        end
    end
end

# Optimize RFOs by combining terms that can be combined
function optimize_operators(ops)
    ops = [op for op in ops if !(op isa NoopOperator)]
    RSmults = [op for op in ops if op isa RealSpaceMultiplication]
    matrix_mults = [op for op in ops if op isa MatrixLocalOperator]
    others = RealFourierOperator[op for op in ops
                                 if !(op isa RealSpaceMultiplication) &&
                                    !(op isa MatrixLocalOperator)]
    if !isempty(RSmults)
        push!(others, RealSpaceMultiplication(RSmults[1].basis,
                                              RSmults[1].kpoint,
                                              sum(op.potential for op in RSmults)))
    end
    if !isempty(matrix_mults)
        push!(others, MatrixLocalOperator(matrix_mults[1].basis,
                                          matrix_mults[1].kpoint,
                                          sum(op.potential for op in matrix_mults)))
    end
    others
end
