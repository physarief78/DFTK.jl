# Included in DFTKFastGaussQuadratureExt. No external package in core postprocessing.

function _screening_voxel_basis(basis,kernel)
    basis.kgrid isa DFTK.MonkhorstPack || error("Screened cell integration needs a regular mesh.")
    B = basis.model.recip_lattice
    size_k = basis.kgrid.kgrid_size
    if kernel isa DFTK.BareCoulomb
        return Matrix(B*Diagonal(1 ./ size_k))
    end
    alignments = [abs(dot(kernel.normal,B[:,i]))/norm(B[:,i]) for i=1:3]
    d = argmax(alignments)
    plane = filter(!=(d),1:3)
    tolerance = 100eps(eltype(basis))
    abs(alignments[d]-1) ≤ tolerance && all(alignments[plane] .≤ tolerance) ||
        error("Slab screening needs a perpendicular vacuum vector.")
    size_k[d] == 1 || error("Use one k point along the slab normal.")
    height = inv(norm(basis.model.inv_lattice[d,:]))
    abs(2kernel.R/height-1) ≤ tolerance || error("Use a half-height slab cutoff.")
    reduce(hcat,[B[:,i]/size_k[i] for i in plane])
end

function _screening_voxel_nodes(V,nodes,weights)
    if size(V,2) == 3
        return [(V*DFTK.Vec3(x,y,z),wx*wy*wz)
                for (x,wx) in zip(nodes,weights) for (y,wy) in zip(nodes,weights)
                for (z,wz) in zip(nodes,weights)]
    end
    [(V[:,1]*x+V[:,2]*y,wx*wy)
     for (x,wx) in zip(nodes,weights) for (y,wy) in zip(nodes,weights)]
end

# Bare slab head: analytically remove its 4πR/|q| singularity.
function _slab_screening_voxel_head(V,R,nodes,weights)
    T = eltype(V)
    area = norm(cross(V[:,1],V[:,2]))
    result = 4T(π)*R*_slab_voxel_singular_integral(V[:,1],V[:,2])/area
    for (q,weight) in _screening_voxel_nodes(V,nodes,weights)
        r = norm(q)
        x = r*R
        regular = x < sqrt(eps(T)) ? -2T(π)*R^2 : 4T(π)*(-expm1(-x)-x)/r^2
        result += weight*regular
    end
    result
end

@doc raw"""
Schur complement for the static long-wavelength cell model:
```math
S=(I-v_b(C+K_{bb}))^{-1}v_b,\quad
l(q)=K_{0b}+q^TB,\quad
w(q)=[v_0(q)^{-1}-K_{00}-q^TAq-l(q)Sl(q)^\dagger]^{-1}.
```
Then W00=w, Wb0=S*l†*w and Wbb=S+S*l†*w*l*S. For an insulator,
the bulk head is `4π/(qᵀεq)` with `ε=I-4π(A+B*S*B†)`. Transform the voxel
by `sqrt(ε)` and reduce its singular integral to faces. The other blocks are
bounded or odd. Slabs instead subtract their analytic bare 1/|q| head.
"""
function DFTK._integrate_screening_voxel(expansion::NamedTuple,
        kernel::Union{DFTK.BareCoulomb,DFTK.SlabTruncatedCoulomb},
        basis::PlaneWaveBasis,n::Integer)
    n ≥ 2 && iseven(n) || throw(ArgumentError("Use an even quadrature order of at least two."))
    (;A,B,C,K,Gs,body) = expansion
    T = real(eltype(A))
    V = T.(_screening_voxel_basis(basis,kernel))
    nodes,weights = gausslegendre(n)
    nodes,weights = T.(nodes/2),T.(weights/2)
    vbody = T.([eval_kernel_fourier(kernel,basis.model.recip_lattice*Gs[i]) for i in body])
    nb = length(body)
    S = (Matrix{Complex{T}}(I,nb,nb)-Diagonal(vbody)*(C+K[2:end,2:end])) \
        Matrix(Diagonal(vbody))
    norm(S-S') ≤ max(T(1e-8),100eps(T))*max(one(T),norm(S)) ||
        error("Static body screening is not Hermitian.")
    S = (S+S')/2
    dielectric = real.(Matrix{Complex{T}}(I,3,3)-4T(π)*(A+B*S*B'))
    dielectric = (dielectric+dielectric')/2
    insulating = norm(K) ≤ 100eps(T)
    W = zeros(Complex{T},nb+1,nb+1)
    W[2:end,2:end] = S
    if insulating && kernel isa DFTK.BareCoulomb
        minimum(eigvals(Symmetric(dielectric))) > 0 ||
            error("The static longitudinal dielectric tensor is not positive.")
        transformed = sqrt(Symmetric(dielectric))*V
        W[1,1] = _coulomb_voxel_head(transformed,nodes,weights)
        J = zeros(T,3,3)
        for (q,weight) in _screening_voxel_nodes(V,nodes,weights)
            J .+= (weight*4T(π)/dot(q,dielectric*q)).*(q*q')
        end
        W[2:end,2:end] .+= S*B'*J*B*S
    else
        subtract_slab = insulating && kernel isa DFTK.SlabTruncatedCoulomb
        subtract_slab && (W[1,1] = _slab_screening_voxel_head(V,kernel.R,nodes,weights))
        for (q,weight) in _screening_voxel_nodes(V,nodes,weights)
            vhead = eval_kernel_fourier(kernel,q)
            l = vec(transpose(q)*B)+K[1,2:end]
            column = S*conj(l)
            denominator = inv(vhead)-real(K[1,1]+dot(q,A*q)+dot(conj(l),column))
            denominator > 0 || error("Unstable long-wavelength static screening at a cell node.")
            w = inv(denominator)
            W[1,1] += weight*(subtract_slab ? w-vhead : w)
            W[2:end,1] .+= weight*w.*column
            W[1,2:end] .+= weight*w.*conj(column)
            W[2:end,2:end] .+= weight*w.*(column*column')
        end
    end
    (;W,dielectric_tensor=insulating && kernel isa DFTK.BareCoulomb ? dielectric : nothing,
      compressibility=-real(K[1,1]))
end
