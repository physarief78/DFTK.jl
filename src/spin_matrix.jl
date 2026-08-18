# Real parametrizations of small Hermitian matrices.
#
# The first generalized Pauli matrix is the identity. The remaining matrices are the
# generalized Gell-Mann matrices, ordered as real off-diagonal, imaginary off-diagonal,
# then diagonal matrices. For C=2 this gives (I, σx, σy, σz).

function generalized_pauli_matrices(::Val{C}, ::Type{T}=Float64) where {C,T}
    CT = Complex{T}
    matrices = SMatrix{C,C,CT,C^2}[]
    push!(matrices, one(SMatrix{C,C,CT,C^2}))

    for j = 2:C, i = 1:j-1
        matrix = zeros(MMatrix{C,C,CT,C^2})
        matrix[i, j] = matrix[j, i] = one(T)
        push!(matrices, SMatrix(matrix))
    end
    for j = 2:C, i = 1:j-1
        matrix = zeros(MMatrix{C,C,CT,C^2})
        matrix[i, j] = -im * one(T)
        matrix[j, i] =  im * one(T)
        push!(matrices, SMatrix(matrix))
    end
    for k = 1:C-1
        matrix = zeros(MMatrix{C,C,CT,C^2})
        normalization = sqrt(T(2) / (T(k) * T(k + 1)))
        for i = 1:k
            matrix[i, i] = normalization
        end
        matrix[k + 1, k + 1] = -T(k) * normalization
        push!(matrices, SMatrix(matrix))
    end
    @assert length(matrices) == C^2
    Tuple(matrices)
end

"""
    hermitian_basis(Val(C), [T])

Orthonormal basis of `C×C` Hermitian matrices. The first matrix is `I / √C`; the
remaining matrices are generalized Gell-Mann matrices divided by `√2`.
"""
function hermitian_basis(::Val{C}, ::Type{T}=Float64) where {C,T}
    pauli = generalized_pauli_matrices(Val(C), T)
    ntuple(Val(C^2)) do i
        pauli[i] / (i == 1 ? sqrt(T(C)) : sqrt(T(2)))
    end
end

"""
    density_matrix(fields)

Convert generalized Pauli fields `(n, m₁, …)` to a Hermitian density matrix. For
`C=2`, this is `½(n I + mx σx + my σy + mz σz)`.
"""
function density_matrix(fields::SVector{N,T}) where {N,T}
    C = isqrt(N)
    C^2 == N || throw(ArgumentError("The field count must be a perfect square."))
    pauli = generalized_pauli_matrices(Val(C), T)
    matrix = fields[1] / C * pauli[1]
    for i = 2:N
        matrix += fields[i] / 2 * pauli[i]
    end
    matrix
end

"""Convert a Hermitian density matrix to generalized Pauli fields `(n, m₁, …)`."""
function pauli_components(matrix::SMatrix{C,C,CT,L}) where {C,CT,L}
    RT = real(CT)
    pauli = generalized_pauli_matrices(Val(C), RT)
    SVector{C^2,RT}(ntuple(i -> real(tr(pauli[i] * matrix)), Val(C^2)))
end

"""
    potential_matrix(fields)

Build the Hermitian matrix dual to generalized Pauli density fields. If `fields` is
`(v, bx, by, bz)`, the result is `v I + bx σx + by σy + bz σz`.
"""
function potential_matrix(fields::SVector{N,T}) where {N,T}
    C = isqrt(N)
    C^2 == N || throw(ArgumentError("The field count must be a perfect square."))
    pauli = generalized_pauli_matrices(Val(C), T)
    sum(i -> fields[i] * pauli[i], 1:N)
end
