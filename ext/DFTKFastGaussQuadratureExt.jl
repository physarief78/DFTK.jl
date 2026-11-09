module DFTKFastGaussQuadratureExt
using FastGaussQuadrature
using DFTK
using DFTK: to_cpu, to_device, norm2, eval_kernel_fourier, VoxelAverage
using LinearAlgebra

DFTK.error_if_kernel_not_available(::Type{<:VoxelAverage}, string) = nothing

function _coulomb_voxel_head(voxel_basis,nodes,weights)
    T = eltype(voxel_basis)
    integral = zero(T)
    for i=1:3
        u_i = voxel_basis[:,i]
        u_j = voxel_basis[:,mod1(i+1,3)]
        u_k = voxel_basis[:,mod1(i+2,3)]
        normal = cross(u_j,u_k)
        area = norm(normal)
        h = abs(dot(u_i,normal))/(2area)
        face_integral = zero(T)
        for (wa,a) in zip(weights,nodes),(wb,b) in zip(weights,nodes)
            r = 0.5f0*u_i+a*u_j+b*u_k
            face_integral += wa*wb/dot(r,r)
        end
        integral += 2h*area*face_integral
    end
    4T(π)*integral/abs(det(voxel_basis))
end

# ∫cell d²p/|p| = ∮boundary (p⋅n)/|p| ds, for a centred parallelogram.
function _slab_voxel_singular_integral(u, v)
    area = norm(cross(u, v))
    integral = zero(eltype(u))
    for (a, b) in ((u, v), (v, u))
        length_b = norm(b)
        h = area/(2length_b)
        s = dot(a, b)/(2length_b)
        integral += 2h*(asinh((s+length_b/2)/h)-asinh((s-length_b/2)/h))
    end
    integral
end

@doc raw"""
For `VoxelAverage(SlabTruncatedCoulomb(...))`, average only over the in-plane
reciprocal cell, with one k point along the perpendicular vacuum vector:
```math
\bar v(G+q)=A_q^{-1}\int_{\mathrm{cell}_{2D}}v(G+q+p)\,d^2p.
```
At the origin subtract `4πR/|p|` and integrate it analytically over the
parallelogram. The remainder is finite and uses 2D Gauss quadrature. This
integrates the bare kernel; q-dependent screening still needs k-mesh convergence.
The vacuum lattice vector must be perpendicular to the slab, R must be its half
height, and q must lie in the slab plane. No 3D voxel is substituted for the slab.
"""
function DFTK.eval_kernel_fourier(average::VoxelAverage{<:DFTK.SlabTruncatedCoulomb},
                                 basis::PlaneWaveBasis{T}, q::Vec3) where {T}
    kernel = average.inner_kernel
    basis.kgrid isa DFTK.MonkhorstPack ||
        error("Slab averaging needs a MonkhorstPack mesh.")
    B = basis.model.recip_lattice
    alignments = [abs(dot(kernel.normal, B[:, i]))/norm(B[:, i]) for i=1:3]
    d = argmax(alignments)
    plane = filter(!=(d), 1:3)
    tolerance = 100eps(T)
    abs(alignments[d]-1) ≤ tolerance && all(alignments[plane] .≤ tolerance) ||
        error("Slab averaging needs a perpendicular vacuum lattice vector.")
    basis.kgrid.kgrid_size[d] == 1 || error("Use one k point along the slab normal.")
    abs(dot(kernel.normal, B*q)) ≤ tolerance*norm(B) ||
        error("q must be in the slab plane.")
    height = inv(norm(basis.model.inv_lattice[d, :]))
    abs(2kernel.R/height-1) ≤ tolerance ||
        error("Use a slab cutoff of half the cell height.")
    average.n_quadrature_points ≥ 2 ||
        throw(ArgumentError("Use at least two quadrature nodes."))
    u, v = [B[:, i]/basis.kgrid.kgrid_size[i] for i in plane]
    area = norm(cross(u, v))
    nodes, weights = gausslegendre(average.n_quadrature_points)
    nodes, weights = T.(nodes/2), T.(weights/2)
    result = zeros(T, basis.fft_size)
    for (i, G) in enumerate(to_cpu(DFTK.G_vectors(basis)))
        p = B*(G+q)
        singular = iszero(p)
        value = singular ? 4T(π)*kernel.R*_slab_voxel_singular_integral(u, v)/area : zero(T)
        for (x, wx) in zip(nodes, weights), (y, wy) in zip(nodes, weights)
            δp = x*u+y*v
            integrand = if singular
                r = norm(δp)
                z = r*kernel.R
                z < sqrt(eps(T)) ? -2T(π)*kernel.R^2 :
                    4T(π)*(-expm1(-z)-z)/r^2
            else
                eval_kernel_fourier(kernel, p+δp)
            end
            value += wx*wy*integrand
        end
        result[i] = value
    end
    to_device(basis.architecture, result)
end

@views function DFTK.eval_kernel_fourier(voxel_average::VoxelAverage,
                                         basis::PlaneWaveBasis{T}, q::Vec3) where {T}
    model  = basis.model
    kernel = voxel_average.inner_kernel

    @assert basis.kgrid isa MonkhorstPack
    kgrid_size = basis.kgrid.kgrid_size

    # Define Voxels as reciprocal cell divided by k-mesh
    voxel_basis = model.recip_lattice * Diagonal(1 ./ Vec3{T}(kgrid_size))

    # Get Gauss-Legendre nodes and weights and scale from [-1, 1] to [-0.5, 0.5]
    nodes_std, weights_std = gausslegendre(voxel_average.n_quadrature_points)
    nodes = T.(nodes_std ./ 2)
    weights = T.(weights_std ./ 2)

    # Precompute all local q vectors and their weights for the voxel
    q_locals = vec([voxel_basis * Vec3(x, y, z) for x in nodes, y in nodes, z in nodes])
    w_locals = vec([wx * wy * wz for wx in weights, wy in weights, wz in weights])

    # Kernel on the full FFT cube (see eval_kernel_fourier); linear index 1 is G=0
    kernel_fourier = zeros(T, basis.fft_size)
    for (iG, G) in enumerate(to_cpu(G_vectors(basis)))
        G_cart = model.recip_lattice * (G+q)

        found_singularity = (iG==1 && iszero(q))

        if found_singularity 
            # === At Singularity (G+q=0) use surface reduction method ===

            # We only do that for the 4π/(G+q)^2 kernel, hence the quadrature below
            # covers the rest if kernel is different from BareCoulomb().

            # Transforms volume integral ∫ 1/G^2 dV to surface integral Σ h * ∫ 1/G^2 dS
            kernel_fourier[iG] = _coulomb_voxel_head(voxel_basis,nodes,weights)
        end

        # === Use smooth 3D Gaussian Quadrature ===
        if norm(kgrid_size .* (G + q)) <= 10
            # assume that interaction kernel varies strongly only among the
            # first 10 (hard coded!) nearest neighbors of the origin voxel;
            # kgrid_size .* (G+q) are the coordinates of G+q in units of the voxel basis
            integral = zero(T)
            for i in 1:length(q_locals)
                G_total = G_cart + q_locals[i]

                if found_singularity
                    # switch temporarily to BigFloat to avoid catastrophic cancellation
                    G_total_big = BigFloat.(G_total)
                    val_big = eval_kernel_fourier(kernel, G_total_big)
                    val_big -= 4big(π) / norm2(G_total_big)
                    val = T(val_big)
                else
                    val = eval_kernel_fourier(kernel, G_total)
                end

                integral += w_locals[i] * val
            end
            kernel_fourier[iG] += integral
        else
            # For G vectors far from the origin, the interaction kernel is practically flat
            # over the voxel. Bypassing the 1728-point quadrature and using the exact
            # midpoint saves enormous CPU time.
            kernel_fourier[iG] += eval_kernel_fourier(kernel, G_cart)
        end
    end

    to_device(basis.architecture, kernel_fourier)
end

include("rpa_optical_voxel.jl")

end
