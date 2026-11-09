@testitem "BSE: Hermitian population factors and Wannier estimate" tags=[:bse] begin
    using DFTK
    using LinearAlgebra
    D = [0.5, 0.7]
    Kx = [0.01 0.02; 0.02 0.01]
    Kd = [0.1 0.03; 0.03 0.1]
    weights = [1.0, sqrt(0.5)]
    H = assemble_bse(D, Kx, Kd, weights)
    @test H[1, 1] ≈ 0.42
    @test H[2, 2] ≈ 0.66
    @test H[1, 2] ≈ sqrt(0.5)*0.01
    @test minimum(eigvals(Hermitian(H))) < minimum(D)
    @test_throws ErrorException assemble_bse(D, [0.0 1; 0 0], Kd, weights)
    @test_throws ArgumentError assemble_bse(D, Kx, Kd, [-1.0, 1.0])
    estimate = wannier_mott(1.0, 1.0, 2.0)
    @test estimate.binding_energy == 0.0625
    @test estimate.radius == 4
end

@testitem "BSE: finite-q kernel relabelling on a k mesh" tags=[:bse] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    model = model_atomic(TestCases.silicon.lattice, TestCases.silicon.atoms,
                          TestCases.silicon.positions; symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 1, 1])
    ρ = guess_density(basis)
    ham = Hamiltonian(basis; ρ)
    F = [eigen(Hermitian(Matrix(h))) for h in ham.blocks]
    data = (; basis, ρ, ψ=[f.vectors[:, 1:8] for f in F], eigenvalues=[f.values[1:8] for f in F],
             occupation=[vcat(fill(2.0, 4), zeros(4)) for _ in F])
    function screening(q)
        Gs = DFTK._response_vectors(basis, q, 0.3)
        n = length(Gs)
        (; basis, q, ω=[0.0], η=0.0, Gs, optical_limit=false,
          W=reshape(Matrix{ComplexF64}(I, n, n)*0.02, n, n, 1))
    end
    bse = compute_bse(data; valence=1:4, conduction=5:8, screening, Ecut_exchange=0.3)
    @test bse.H ≈ bse.H' atol=1e-12
    @test norm(bse.Kd) > 0
    phases = [exp.(im.*collect(0.1:0.1:0.8)), exp.(im.*collect(0.2:0.2:1.6))]
    gauged = merge(data, (; ψ=[ψ*Diagonal(p) for (ψ, p) in zip(data.ψ, phases)]))
    changed = compute_bse(gauged; valence=1:4, conduction=5:8, screening, Ecut_exchange=0.3)
    @test changed.energies ≈ bse.energies
end

@testitem "BSE: IPA limit, partial populations and gauge invariance" tags=[:bse] #=
    =# setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    model = model_atomic(TestCases.silicon.lattice, TestCases.silicon.atoms,
                          TestCases.silicon.positions; symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=ExplicitKpoints([[0.13, 0.07, -0.04]]))
    ρ = guess_density(basis)
    F = eigen(Hermitian(Matrix(Hamiltonian(basis; ρ)[1])))
    data = (; basis, ρ, ψ=[F.vectors[:, 1:8]], eigenvalues=[F.values[1:8]],
             occupation=[vcat(fill(2.0, 4), zeros(4))])
    function zero_screening(q)
        (; basis, q, ω=[0.0], η=0.0, Gs=[Vec3(0, 0, 0)], optical_limit=false,
          W=zeros(ComplexF64, 1, 1, 1))
    end
    ω = [0.0, 0.1, 0.3, 0.6]
    for population in (0.0, 0.2), scissor in (0.0, 0.05)
        light = merge(data, (; occupation=[vcat(fill(2-population, 4), fill(population, 4))]))
        bse = compute_bse(light; valence=1:4, conduction=5:8, screening=zero_screening,
                           Ecut_exchange=1e-12, scissor)
        ipa = compute_dielectric(light, ω; η=0.01, scissor, n_valence=4)
        excitonic = compute_bse_dielectric(light, bse, ω; η=0.01)
        @test excitonic.ε ≈ ipa.ε rtol=1e-10 atol=1e-12
        density = compute_exciton_densities(light, bse, 1)
        @test sum(density.ρelectron)*basis.dvol ≈ 1
        @test sum(density.ρhole)*basis.dvol ≈ 1
    end
    function screening(q)
        Gs = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(0, -1, 0)]
        (; basis, q, ω=[0.0], η=0.0, Gs, optical_limit=false,
          W=reshape(Matrix{ComplexF64}(I, 5, 5)*0.1, 5, 5, 1))
    end
    bse = compute_bse(data; valence=1:4, conduction=5:8, screening, Ecut_exchange=0.5)
    phases = exp.(im.*collect(0.1:0.1:0.8))
    gauged = merge(data, (; ψ=[data.ψ[1]*Diagonal(phases)]))
    changed = compute_bse(gauged; valence=1:4, conduction=5:8, screening, Ecut_exchange=0.5)
    @test changed.energies ≈ bse.energies
    @test compute_bse_dielectric(gauged, changed, ω; η=0.01).ε ≈
          compute_bse_dielectric(data, bse, ω; η=0.01).ε rtol=1e-9 atol=1e-10
    allspace = ones(basis.fft_size)
    character = compute_exciton_character(data, bse, 1; electron_mask=allspace, hole_mask=allspace)
    @test character.joint ≈ 1
    @test character.hole_probability ≈ 1
    @test character.conditional_electron_fraction ≈ 1
end
