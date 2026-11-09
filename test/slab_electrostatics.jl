@testitem "Slab Coulomb: analytic limits" tags=[:slab] begin
    using DFTK
    using LinearAlgebra
    kernel = SlabTruncatedCoulomb(10.0)
    @test isinf(DFTK.eval_kernel_fourier(kernel, zeros(3)))
    @test DFTK.eval_kernel_fourier(kernel, [0.2, 0, 0]) ≈ 4π/0.2^2*(1-exp(-2))
    @test DFTK.eval_kernel_fourier(kernel, [0, 0, π/10]) ≈ 8π/(π/10)^2
    @test DFTK.eval_kernel_fourier(kernel, [0, 0, 2π/10]) ≈ 0 atol=1e-12
    @test DFTK.eval_kernel_fourier(kernel, [1e-8, 0, 0])*1e-8 ≈ 40π rtol=1e-7
    p = [0.3, 0.4, 0.17]
    expected = 4π/sum(abs2, p)*(1+exp(-5)*(p[3]/0.5*sin(10p[3])-cos(10p[3])))
    @test DFTK.eval_kernel_fourier(kernel, p) ≈ expected
    @test_throws ArgumentError SlabTruncatedCoulomb(-1.0)
end

@testitem "Slab Coulomb: two-dimensional singularity integral" tags=[:slab] begin
    using DFTK, FastGaussQuadrature, LinearAlgebra
    extension = Base.get_extension(DFTK, :DFTKFastGaussQuadratureExt)
    integral = extension._slab_voxel_singular_integral
    @test integral([1.0, 0, 0], [0, 1.0, 0]) ≈ 4asinh(1)
    @test integral([2.0, 0, 0], [0, 2.0, 0]) ≈ 8asinh(1)
    model = Model(diagm([8.0, 9.0, 20.0]), fill(ElementCoulomb(:H), 2),
                  [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]; terms=[Kinetic()], symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=0.2, kgrid=[4, 6, 1], fft_size=(5, 5, 7))
    kernel = SlabTruncatedCoulomb(basis)
    q0 = Vec3(0.0, 0, 0)
    averaged = DFTK.eval_kernel_fourier(VoxelAverage(kernel; n_quadrature_points=48), basis, q0)
    @test all(isfinite, averaged)
    @test averaged[1] > 0
    # Independent Duffy quadrature: four triangles, p=s*((1-t)*a+t*b),
    # whose Jacobian s removes the 1/|p| singularity without subtraction.
    u, v = model.recip_lattice[:, 1]/4, model.recip_lattice[:, 2]/6
    corners = [(u+v)/2, (-u+v)/2, (-u-v)/2, (u-v)/2]
    nodes, weights = gausslegendre(64)
    nodes, weights = (nodes.+1)/2, weights/2
    reference = let value=0.0
        for i=1:4
            a, b = corners[i], corners[mod1(i+1, 4)]
            for (s, ws) in zip(nodes, weights), (t, wt) in zip(nodes, weights)
                value += ws*wt*s*norm(cross(a, b))*
                          DFTK.eval_kernel_fourier(kernel, s*((1-t)*a+t*b))
            end
        end
        value/norm(cross(u, v))
    end
    @test averaged[1] ≈ reference rtol=2e-6
    coarse = DFTK.eval_kernel_fourier(VoxelAverage(kernel; n_quadrature_points=12), basis, q0)
    @test abs(averaged[1]-reference) < abs(coarse[1]-reference)
    θ = 0.41
    rotation = [1.0 0 0; 0 cos(θ) -sin(θ); 0 sin(θ) cos(θ)]
    rotated_model = Model(model; lattice=rotation*model.lattice)
    rotated_basis = PlaneWaveBasis(basis; model=rotated_model)
    rotated = SlabTruncatedCoulomb(rotated_basis)
    rotated_values = DFTK.eval_kernel_fourier(VoxelAverage(rotated; n_quadrature_points=48),
                                             rotated_basis, q0)
    @test rotated_values ≈ averaged rtol=1e-12
    @test_throws ErrorException DFTK.eval_kernel_fourier(VoxelAverage(kernel), basis,
                                                        Vec3(0.0, 0, 0.1))
    @test_throws ErrorException DFTK.eval_kernel_fourier(
        VoxelAverage(SlabTruncatedCoulomb(9.0)), basis, q0)
end

@testitem "Dipole correction: energy, potential, forces and kernel" tags=[:slab] begin
    using DFTK
    using LinearAlgebra
    lattice = diagm([8.0, 8.0, 20.0])
    atoms = fill(ElementCoulomb(:H), 2)
    positions = [[0.5, 0.5, 0.4], [0.5, 0.5, 0.6]]
    model = Model(lattice, atoms, positions; terms=[Kinetic(), DipoleCorrection()],
                  symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=0.5, kgrid=[1, 1, 1], fft_size=(9, 9, 32))
    term = basis.terms[2]
    ρ = zeros(basis.fft_size..., 1)
    ρ[:, :, 9, 1] .= 1/(81basis.dvol)
    ρ[:, :, 25, 1] .= 1/(81basis.dvol)
    result = DFTK.ene_ops(term, basis, nothing, nothing; ρ)
    @test result.E ≈ 0 atol=1e-24
    @test maximum(abs, only(result.ops).potential) < 1e-12
    ρ[:, :, 9, 1] .*= 1.2
    ρ[:, :, 25, 1] .*= 0.8
    p = DFTK._slab_dipole(term, basis, ρ)
    @test p ≈ 2.0
    result = DFTK.ene_ops(term, basis, nothing, nothing; ρ)
    @test result.E ≈ 2π*p^2/det(lattice)
    V = only(result.ops).potential
    @test V[1, 1, 25]-V[1, 1, 9] ≈ -4π*p/det(lattice)*10
    δρ = zeros(size(ρ))
    δρ[:, :, 13, 1] .= 0.02
    δρ[:, :, 21, 1] .= -0.02
    h = 1e-5
    ep = DFTK.ene_ops(term, basis, nothing, nothing; ρ=ρ+h*δρ)
    em = DFTK.ene_ops(term, basis, nothing, nothing; ρ=ρ-h*δρ)
    @test (ep.E-em.E)/(2h) ≈ sum(V .* total_density(δρ))*basis.dvol rtol=1e-9
    applied = apply_kernel(term, basis, δρ).δVρ
    @test applied[:, :, :, 1] ≈ (only(ep.ops).potential-only(em.ops).potential)/(2h) #=
        =# rtol=1e-8
    @test reshape(DFTK.compute_kernel(term, basis)*vec(δρ), size(ρ)) ≈ applied
    force = DFTK.compute_forces(term, basis, nothing, nothing; ρ)[1][3]
    function displaced_energy(δ)
        shifted = [positions[1]+[0, 0, δ], positions[2]]
        m = Model(model; positions=shifted)
        b = PlaneWaveBasis(basis; model=m)
        DFTK.ene_ops(b.terms[2], b, nothing, nothing; ρ).E
    end
    @test force ≈ -(displaced_energy(h)-displaced_energy(-h))/(2h) rtol=1e-8
    @test_throws ErrorException apply_kernel(term, basis, δρ; q=Vec3(0.1, 0, 0))
    @test_throws ErrorException DFTK.compute_dynmat(term, basis, nothing, nothing)
end

@testitem "Electrostatic analysis: quadrature, Hirshfeld and cube" tags=[:slab] begin
    using DFTK
    using LinearAlgebra
    model = Model(diagm([6.0, 7.0, 12.0]), [ElementCoulomb(:H), ElementCoulomb(:H)],
                  [[0.2, 0.3, 0.4], [0.7, 0.8, 0.6]];
                  terms=[Kinetic()], symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=[1, 1, 1])
    ρ = guess_density(basis)
    data = (; basis, ρ)
    profile = planar_average(basis, total_density(ρ))
    @test sum(profile.values)*profile.spacing*profile.area ≈ sum(ρ)*basis.dvol
    Δρ = density_difference(data, [data])
    @test iszero(Δρ.Δρ)
    @test Δρ.excess_electrons == 0
    cumulative = cumulative_density(basis, total_density(ρ))
    @test last(cumulative.electrons) ≈ sum(ρ)*basis.dvol
    hirshfeld = compute_hirshfeld(data; method=ValenceDensityAuto())
    @test sum(hirshfeld.populations) ≈ sum(ρ)*basis.dvol
    @test maximum(abs, hirshfeld.charges) < 1e-8
    mktempdir() do dir
        path = write_cube(joinpath(dir, "density.cube"), basis, total_density(ρ))
        lines = readlines(path)
        @test parse(Int, first(split(lines[3]))) == 2
        values = parse.(Float64, split(join(lines[9:end], " ")))
        @test length(values) == prod(basis.fft_size)
        @test sum(values)*basis.dvol ≈ sum(ρ)*basis.dvol rtol=1e-6
    end
    @test_throws DimensionMismatch planar_average(basis, zeros(2, 2, 2))
end
