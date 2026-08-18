# Densities (and potentials) are represented by real arrays `ρ[ix, iy, iz, iρ]`.
# Collinear densities retain their `(ρα, ρβ)` fields. Full-spin densities use Pauli
# fields `(n, mx, my, mz)`, representing `ρmat = (n I + m⋅σ) / 2` pointwise.

function _density_local_storage(basis, ::Type{Tρ}, ::Type{Tψ}, ::Val{C}) where {Tρ,Tψ,C}
    ρ = zeros_like(G_vectors(basis), Tρ, basis.fft_size...,
                   n_density_components(basis.model))
    ψ_reals = ntuple(Val(C)) do _
        zeros_like(G_vectors(basis), complex(Tψ), basis.fft_size...)
    end
    (; ρ, ψ_reals)
end

@views function _accumulate_density!(ρ, ::Val{1}, spin::Integer, weight, ψ_reals)
    ρ[:, :, :, spin] .+= weight .* abs2.(only(ψ_reals))
    ρ
end

@views function _accumulate_density!(ρ, ::Val{2}, spin::Integer, weight, ψ_reals)
    @assert spin == 1
    ψ1, ψ2 = ψ_reals
    cross = conj.(ψ1) .* ψ2
    ρ[:, :, :, 1] .+= weight .* (abs2.(ψ1) .+ abs2.(ψ2))
    ρ[:, :, :, 2] .+= 2weight .* real.(cross)
    ρ[:, :, :, 3] .+= 2weight .* imag.(cross)
    ρ[:, :, :, 4] .+= weight .* (abs2.(ψ1) .- abs2.(ψ2))
    ρ
end

@views function _accumulate_density!(ρ, ::Val{C}, spin::Integer, weight,
                                     ψ_reals) where {C}
    @assert spin == 1
    pauli = generalized_pauli_matrices(Val(C), real(eltype(first(ψ_reals))))
    for i = 1:C^2
        contribution = sum(1:C) do σ
            sum(1:C) do τ
                pauli[i][σ, τ] .* conj.(ψ_reals[σ]) .* ψ_reals[τ]
            end
        end
        ρ[:, :, :, i] .+= weight .* real.(contribution)
    end
    ρ
end

"""
    compute_density(basis::PlaneWaveBasis, ψ::AbstractVector, occupation::AbstractVector)

Compute the density for a wave function `ψ` discretized on the plane-wave
grid `basis`, where the individual k-points are occupied according to `occupation`.
`ψ` should be one coefficient matrix per ``k``-point.
It is possible to ask only for occupations higher than a certain level to be computed by
using an optional `occupation_threshold`. By default all occupation numbers are considered.
"""
@views @timing function compute_density(basis::PlaneWaveBasis{T,VT}, ψ, occupation;
                                        occupation_threshold=zero(T), label=:ρ) where {T,VT}
    # Occupation should be on the CPU as we are going to be doing scalar indexing.
    occupation = [to_cpu(oc) for oc in occupation]
    mask_occ = [findall(occnk -> abs(occnk) ≥ occupation_threshold, occk)
                for occk in occupation]

    function allocate_local_storage()
        # The types were moved inside here to avoid a type instability,
        # as it seems that captures over types do not get specialized!
        Tρ = promote_type(T,  real(eltype(ψ[1])))
        # Note, that we special-case Tψ, since when T is Dual and eltype(ψ[1]) is not
        # (e.g. stress calculation), then only the normalisation factor introduces
        # dual numbers, but not yet the FFT
        Tψ = promote_type(VT, real(eltype(ψ[1])))

        _density_local_storage(basis, Tρ, Tψ, Val(n_components(basis.model)))
    end
    # We split the total iteration range (ik, n) in chunks, and parallelize over them.
    range = [(ik, n) for ik = 1:length(basis.kpoints) for n = mask_occ[ik]]

    storages = parallel_loop_over_range(range; allocate_local_storage) do kn, storage
        (ik, n) = kn
        kpt = basis.kpoints[ik]
        ifft_components!(storage.ψ_reals, basis, kpt, ψ[ik], n; normalize=false)
        weight = (occupation[ik][n] * basis.kweights[ik]
                  * basis.fft_grid.ifft_normalization^2)
        _accumulate_density!(storage.ρ, Val(length(storage.ψ_reals)), kpt.spin,
                             weight, storage.ψ_reals)

    end
    ρ = sum(storage -> storage.ρ, storages)

    mpi_sum!(ρ, basis.comm_kpts)
    ρ = symmetrize_ρ(basis, ρ)

    # There can always be small negative densities, e.g. due to numerical fluctuations
    # in a vacuum region, so put some tolerance even if occupation_threshold == 0
    negtol = max(sqrt(eps(T)), 10occupation_threshold)
    if mpi_master(basis.comm_kpts)
        # The magnetization fields of a full-spin density may legitimately be negative;
        # only its total-density field is subject to this check. Keep the legacy path on
        # the complete array both to validate every collinear channel and to avoid
        # materializing a total-density cube in the one-component hot path.
        min_ρ = if basis.model.spin_polarization == :full
            minimum(view(ρ, :, :, :, 1))
        else
            minimum(ρ)
        end
        min_ρ < -negtol && @warn("Negative $label detected", min_ρ)
    end

    ρ
end

# Zero-cost compatibility path for the established matrix-per-k-point representation.
# Component-carrying orbitals are three-dimensional and therefore stay on the generic
# Pauli-field implementation above.
@views @timing function compute_density(basis::PlaneWaveBasis{T,VT},
                                        ψ::AbstractVector{<:AbstractMatrix}, occupation;
                                        occupation_threshold=zero(T), label=:ρ) where {T,VT}
    n_components(basis.model) == 1 || throw(ArgumentError(
        "Full-spin orbitals require an explicit component axis."))
    occupation = [to_cpu(oc) for oc in occupation]
    mask_occ = [findall(occnk -> abs(occnk) ≥ occupation_threshold, occk)
                for occk in occupation]

    function allocate_local_storage()
        Tρ = promote_type(T, real(eltype(ψ[1])))
        Tψ = promote_type(VT, real(eltype(ψ[1])))
        (; ρ=zeros_like(G_vectors(basis), Tρ, basis.fft_size...,
                        basis.model.n_spin_components),
           ψnk_real=zeros_like(G_vectors(basis), complex(Tψ), basis.fft_size...))
    end
    range = [(ik, n) for ik = 1:length(basis.kpoints) for n = mask_occ[ik]]
    storages = parallel_loop_over_range(range; allocate_local_storage) do kn, storage
        (ik, n) = kn
        kpt = basis.kpoints[ik]
        ifft!(storage.ψnk_real, basis, kpt, ψ[ik][:, n]; normalize=false)
        storage.ρ[:, :, :, kpt.spin] .+= (occupation[ik][n] * basis.kweights[ik]
                                          * basis.fft_grid.ifft_normalization^2
                                          .* abs2.(storage.ψnk_real))
    end
    ρ = sum(storage -> storage.ρ, storages)

    mpi_sum!(ρ, basis.comm_kpts)
    ρ = symmetrize_ρ(basis, ρ)
    negtol = max(sqrt(eps(T)), 10occupation_threshold)
    if mpi_master(basis.comm_kpts)
        min_ρ = minimum(ρ)
        min_ρ < -negtol && @warn("Negative $label detected", min_ρ)
    end
    ρ
end

# Variation in density corresponding to a variation in the orbitals and occupations.
@views @timing function compute_δρ(basis::PlaneWaveBasis{T}, ψ, δψ, occupation,
                                   δoccupation=zero.(occupation);
                                   occupation_threshold=zero(T), q=zero(Vec3{T})) where {T}
    n_components(basis.model) == 1 || error(
        "Density response is not yet implemented for full spin polarization.")
    Tψ = promote_type(T, eltype(ψ[1]))
    # δρ is expected to be real when computations are not phonon-related.
    Tδρ = iszero(q) ? real(Tψ) : Tψ
    real_qzero = iszero(q) ? real : identity

    # Note: This routine assumes the existence of time-reversal symmetry
    #       in the model/discretisation for q!=0, see also the comment in postprocess/phonon.jl.
    @assert iszero(q) || !breaks_time_reversal_symmetry(basis)

    # occupation should be on the CPU as we are going to be doing scalar indexing.
    occupation = [to_cpu(oc) for oc in occupation]
    mask_occ = [findall(occnk -> abs(occnk) ≥ occupation_threshold, occk)
                for occk in occupation]

    function allocate_local_storage()
        (; δρ=zeros_like(G_vectors(basis), Tδρ, basis.fft_size...,
                         n_density_components(basis.model)),
          ψnk_real=zeros_like(G_vectors(basis), Tψ, basis.fft_size...),
         δψnk_real=zeros_like(G_vectors(basis), Tψ, basis.fft_size...))
    end
    range = [(ik, n) for ik = 1:length(basis.kpoints) for n = mask_occ[ik]]

    # The variation of the orbital ψ_k defined in the basis ℬ_k is δψ_{[k+q]} in ℬ_{[k+q]},
    # where [k+q] is equivalent to the basis k+q (see find_equivalent_kpt).
    # The perturbation of the density
    #   |ψ_{n,k}|² is 2 ψ_{n,k} * δψ_{n,k+q}.
    # Hence, we first get the δψ_{[k+q]} as δψ_{k+q}…
    δψ_plus_q = transfer_blochwave_equivalent_to_actual(basis, δψ, q)
    storages = parallel_loop_over_range(range; allocate_local_storage) do kn, storage
        (ik, n) = kn

        kpt = basis.kpoints[ik]
        ifft!(storage.ψnk_real, basis, kpt, ψ[ik][:, n]; normalize=false)
        # … and then we compute the real Fourier transform in the adequate basis.
        ifft!(storage.δψnk_real, basis, δψ_plus_q[ik].kpt, δψ_plus_q[ik].ψk[:, n]; normalize=false)
        # use unnormalized plans for extra speed, normalize at the end
        ifft_normalization = basis.fft_grid.ifft_normalization

        # For the (non-trivial) explanation of why we don't take real parts when q!=0,
        # see comment in postprocess/phonon.jl
        storage.δρ[:, :, :, kpt.spin] .+= ifft_normalization^2 .* real_qzero.(
            2 .* occupation[ik][n]  .* basis.kweights[ik] .* conj.(storage.ψnk_real)
                                                          .* storage.δψnk_real
              .+ δoccupation[ik][n] .* basis.kweights[ik] .* abs2.(storage.ψnk_real))

    end
    δρ = sum(getfield.(storages, :δρ))

    mpi_sum!(δρ, basis.comm_kpts)
    symmetrize_ρ(basis, δρ)
end

@views @timing function compute_kinetic_energy_density(basis::PlaneWaveBasis, ψ, occupation)
    n_components(basis.model) == 1 || error(
        "Kinetic-energy densities are not yet implemented for full spin polarization.")
    T = promote_type(eltype(basis), real(eltype(ψ[1])))
    τ = similar(ψ[1], T, (basis.fft_size..., basis.model.n_spin_components))
    τ .= 0
    dαψnk_real = zeros_like(G_vectors(basis), complex(T), basis.fft_size...)
    occupation = [to_cpu(oc) for oc in occupation]
    for (ik, kpt) in enumerate(basis.kpoints)
        G_plus_k = [map(p -> p[α], Gplusk_vectors_cart(basis, kpt)) for α = 1:3]
        for n = 1:size(ψ[ik], 2), α = 1:3
            ifft!(dαψnk_real, basis, kpt, im .* G_plus_k[α] .* ψ[ik][:, n])
            @. τ[:, :, :, kpt.spin] += occupation[ik][n] * basis.kweights[ik] / 2 * abs2(dαψnk_real)
        end
    end
    mpi_sum!(τ, basis.comm_kpts)
    symmetrize_ρ(basis, τ)
end

# Variation in kinetic energy density corresponding to a variation in the orbitals and occupations.
@views @timing function compute_δτ(basis::PlaneWaveBasis{T}, ψ, δψ, occupation,
                                   δoccupation=zero.(occupation);
                                   q=zero(Vec3{T})) where {T}
    Tψ = promote_type(T, eltype(ψ[1]))
    # δτ is expected to be real when computations are not phonon-related.
    Tδτ = iszero(q) ? real(Tψ) : Tψ
    real_qzero = iszero(q) ? real : identity

    # occupation should be on the CPU as we are going to be doing scalar indexing.
    occupation = [to_cpu(oc) for oc in occupation]

    δτ=zeros_like(G_vectors(basis), Tδτ, basis.fft_size..., basis.model.n_spin_components)
    dαψnk_real=zeros_like(G_vectors(basis), Tψ, basis.fft_size...)
    dαδψnk_real=zeros_like(G_vectors(basis), Tψ, basis.fft_size...)

    # The perturbation of the kinetic energy density (see also compute_δρ above) is
    #   1/2 * |∇ψ_{n,k}|² is ∇ψ_{n,k}* ∇δψ_{n,k+q}.
    # Note that we need (G+k) for the first gradient and (G+k+q) for the second one.
    δψ_plus_q = transfer_blochwave_equivalent_to_actual(basis, δψ, q)

    for (ik, kpt) in enumerate(basis.kpoints)
        G_plus_k        = [map(p -> p[α], Gplusk_vectors_cart(basis, kpt)) for α = 1:3]
        G_plus_k_plus_q = [map(p -> p[α], Gplusk_vectors_cart(basis, δψ_plus_q[ik].kpt)) for α = 1:3]
        for n = 1:size(ψ[ik], 2), α = 1:3
            ifft!(dαψnk_real, basis, kpt, im .* G_plus_k[α] .* ψ[ik][:, n])
            ifft!(dαδψnk_real, basis, δψ_plus_q[ik].kpt, im .* G_plus_k_plus_q[α] .* δψ_plus_q[ik].ψk[:, n])
            @. δτ[:, :, :, kpt.spin] += real_qzero.(
                # note: 1/2 factor from the kinetic energy cancels out the 2 factor from the square
                   occupation[ik][n]  .* basis.kweights[ik] .* conj.(dαψnk_real) .* dαδψnk_real
                .+ δoccupation[ik][n] .* basis.kweights[ik] .* abs2.(dαψnk_real) ./ 2)
        end
    end

    mpi_sum!(δτ, basis.comm_kpts)
    symmetrize_ρ(basis, δτ)
end

"""
Von Weizsäcker kinetic energy density, which is exact for one-electron systems
(i.e. if only one spin channels is occupied or both spin channels singly occupied).
"""
@timing function von_weizsaecker_kinetic_energy_density(basis::PlaneWaveBasis,
                                                        ρ::AbstractArray{T};
                                                        ε_regularization=10eps(T)) where {T}
    n_components(basis.model) == 1 || error(
        "Von Weizsäcker densities are not yet implemented for full spin polarization.")
    ρ_fourier = fft(basis, ρ)
    τ = zero(ρ)
    G = [map(G -> G[α], G_vectors_cart(basis)) for α = 1:3]
    for σ = 1:basis.model.n_spin_components, α = 1:3
        ∇ρ_ασ = ifft(basis, im .* G[α] .* @view ρ_fourier[:, :, :, σ])
        τ[:, :, :, σ] += abs2.(∇ρ_ασ)
    end

    # The maximum regularises for cases where ρ is small to avoid an Inf or NaN appearing.
    τ = @. τ / (8max(ρ, ε_regularization))

    τ
end


"""
    total_density(ρ)

Return the total charge-density field. For full-spin input this is the first Pauli field.
"""
function total_density(ρ)
    n_fields = size(ρ, 4)
    if n_fields == 2
        dropdims(sum(ρ; dims=4); dims=4)
    else
        copy(@view ρ[:, :, :, 1])
    end
end

"""
    spin_density(ρ)

Return the collinear spin density. For full-spin input this returns `mz`; use
[`magnetization`](@ref) to obtain all three Cartesian fields.
"""
@views function spin_density(ρ)
    if size(ρ, 4) == 2
        ρ[:, :, :, 1] - ρ[:, :, :, 2]
    elseif size(ρ, 4) == 4
        copy(ρ[:, :, :, 4])
    else
        zero(ρ[:, :, :])
    end
end

"""
    magnetization(basis, ρ)

Return the Cartesian magnetization fields as an `(fft_size..., 3)` array.
"""
@views function magnetization(basis, ρ)
    T = eltype(ρ)
    if basis.model.spin_polarization == :full
        copy(ρ[:, :, :, 2:4])
    elseif basis.model.spin_polarization == :collinear
        zeros_xy = zeros_like(G_vectors(basis), T, basis.fft_size..., 2)
        cat(zeros_xy, spin_density(ρ); dims=Val(4))
    else
        zeros_like(G_vectors(basis), T, basis.fft_size..., 3)
    end
end

"""Construct the legacy one- or two-field density from total and collinear spin fields."""
function ρ_from_total_and_spin(ρtot, ρspin=nothing)
    if isnothing(ρspin)
        # Val used to ensure inferability
        cat(ρtot; dims=Val(4))  # copy for consistency with other case
    else
        cat((ρtot .+ ρspin) ./ 2,
            (ρtot .- ρspin) ./ 2; dims=Val(4))
    end
end

"""
    ρ_from_total_and_magnetization(ρtot, magnetization)

Construct full-spin Pauli fields `(n, mx, my, mz)` from a total-density cube and an
`(fft_size..., 3)` Cartesian magnetization array.
"""
function ρ_from_total_and_magnetization(ρtot, magnetization)
    @assert size(magnetization) == (size(ρtot)..., 3)
    cat(ρtot, magnetization; dims=Val(4))
end

function ρ_from_total(basis, ρtot::AbstractArray{T}) where {T}
    if basis.model.spin_polarization in (:none, :spinless)
        ρspin = nothing
    elseif basis.model.spin_polarization == :full
        magnetization = zeros_like(G_vectors(basis), T, basis.fft_size..., 3)
        return ρ_from_total_and_magnetization(ρtot, magnetization)
    else
        ρspin = zeros_like(G_vectors(basis), T, basis.fft_size...)
    end
    ρ_from_total_and_spin(ρtot, ρspin)
end


struct DensityMatrixView{C,N,T,A<:AbstractArray{T,4}} <:
       AbstractArray{SMatrix{C,C,Complex{T},N},3}
    fields::A
end
Base.size(view::DensityMatrixView) = size(view.fields)[1:3]
Base.IndexStyle(::Type{<:DensityMatrixView}) = IndexCartesian()
function Base.getindex(view::DensityMatrixView{C,N}, i::Int, j::Int,
                       k::Int) where {C,N}
    fields = SVector{N}(ntuple(l -> view.fields[i, j, k, l], Val(N)))
    density_matrix(fields)
end

"""
    density_matrices(ρ)

Return a lazy three-dimensional view of the pointwise Hermitian density matrices encoded
by a `C²`-field density array.
"""
function density_matrices(ρ::AbstractArray{T,4}) where {T}
    C = isqrt(size(ρ, 4))
    C^2 == size(ρ, 4) || throw(ArgumentError("Density field count must be C²."))
    DensityMatrixView{C,C^2,T,typeof(ρ)}(ρ)
end

"""Encode pointwise Hermitian `matrices` into the generalized Pauli fields of `ρ`."""
function from_density_matrices!(ρ::AbstractArray{T,4}, matrices) where {T}
    size(ρ)[1:3] == size(matrices) || throw(DimensionMismatch("Grid sizes do not match."))
    C = isqrt(size(ρ, 4))
    C^2 == size(ρ, 4) || throw(ArgumentError("Density field count must be C²."))
    # Broadcast rather than index elementwise, such that this also works on the device.
    fields = pauli_components.(matrices)
    for i = 1:C^2
        @views ρ[:, :, :, i] .= getindex.(fields, i)
    end
    ρ
end

#
# Generalised density representation
#
# Many DFT functionals rely on the Hoffmann-Ostenhoff inequality τW(ρ) ≤ τ, i.e. that the
# von Weizsäcker kinetic energy density is strictly a lower bound to the kinetic energy density.
# This is satisfied if ρ and τ are built from the same orbitals, but may not be true if we
# take (convex) combinations of the vector (ρ, τ). This is why we represent the packed
# density as (ρ, τ_excess) where τ_excess = τ - τW(ρ).
#
# TODO: When we do the state refactor we should think how to deal with this;
#       - Probably a ComponentArray or some form of custom struct is reasonable here
#       - Do we want to compute the τW implicitly in here ? It's not a very cheap operation
#         (i.e. not order-1) so that may be unexpected for such an operation. On the other
#         hand we will probably not get around to have some form of additional computation
#         that happens upon the construction of these "densities".
pack_gdensity(basis::PlaneWaveBasis, ρ::AbstractArray, τ::Nothing) = ρ
pack_gdensity(basis, ρ::AbstractArray, τ::AbstractArray) = cat(ρ, τ; dims=Val(4))
function split_gdensity(basis::PlaneWaveBasis, x::AbstractArray{T, 4}) where {T}
    # TODO: This breaks when non-collinear spin is implemented
    n_density = n_density_components(basis.model)
    @assert size(x, 4) == n_density || size(x, 4) == 2n_density
    if size(x, 4) == 2n_density
        ρ = @view x[:, :, :,           1:n_density]
        τ = @view x[:, :, :, n_density+1:end      ]
        (ρ, τ)
    else
        (x, nothing)
    end
end
