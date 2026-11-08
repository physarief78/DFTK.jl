@testitem "Photo occupations: quasi-Fermi constraints and limits" tags=[:photo] begin
    using DFTK
    using LinearAlgebra
    model = Model(8.0Matrix(I,3,3);n_electrons=4,temperature=0.02,
                  smearing=Smearing.FermiDirac(),terms=[Kinetic()],symmetries=false)
    basis = PlaneWaveBasis(model;Ecut=1,kgrid=[2,1,1])
    ε = [[-0.8,-0.2,0.3,0.7],[-0.7,-0.1,0.4,0.8]]
    for n_eh in (0.0,0.05,0.5,2.0,4.0)
        result = DFTK.compute_occupation(basis,ε,QuasiFermi(2,n_eh);tol_n_elec=1e-12)
        @test DFTK.weighted_ksum(basis,[sum(f[1:2]) for f in result.occupation]) ≈ 4-n_eh #=
            =# atol=1e-12
        @test DFTK.weighted_ksum(basis,[sum(f[3:4]) for f in result.occupation]) ≈ n_eh #=
            =# atol=1e-12
        @test DFTK.weighted_ksum(basis,sum.(result.occupation)) ≈ 4 atol=1e-12
        @test all(f -> all(x -> 0 ≤ x ≤ 2,f),result.occupation)
    end
    result = DFTK.compute_occupation(basis,ε,QuasiFermi(2,0.2))
    derivative = compute_occupation_derivative(basis,ε,result.εF)
    @test all(f -> all(≤(0),f),derivative)
    h = 1e-6
    fp = DFTK.compute_occupation(basis,[εk.+h for εk in ε],result.εF).occupation
    fm = DFTK.compute_occupation(basis,[εk.-h for εk in ε],result.εF).occupation
    @test derivative ≈ (fp-fm)/(2h) rtol=1e-7
    zero_model = Model(model.lattice;n_electrons=4,temperature=0.0,
                        terms=[Kinetic()],symmetries=false)
    zeroT = PlaneWaveBasis(basis;model=zero_model)
    @test DFTK.compute_occupation(zeroT,ε,QuasiFermi(2,0)).occupation ==
          DFTK.compute_occupation(zeroT,ε).occupation
    @test_throws ErrorException DFTK.compute_occupation(zeroT,ε,QuasiFermi(2,0.1))
    @test_throws ErrorException DFTK.compute_occupation(basis,ε,QuasiFermi(2,5))
    @test_throws ErrorException DFTK.compute_occupation(basis,
        [[-0.8,0.5,0.6,0.7],[-0.7,-0.1,0.4,0.8]],QuasiFermi(2,0.1))
    @test_throws ArgumentError QuasiFermi(2,-0.1)
end

@testitem "Photo occupations: fixed ΔSCF and SCF entropy" tags=[:photo] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    using .TestCases: silicon
    model = model_DFT(silicon.lattice,silicon.atoms,silicon.positions;
                      functionals=LDA(),temperature=0.01,
                      smearing=Smearing.FermiDirac(),symmetries=false)
    basis = PlaneWaveBasis(model;Ecut=5,kgrid=[1,1,1])
    bands = FixedBands(;n_bands_converge=12,n_bands_compute=12,occupation_threshold=1e-12)
    result = self_consistent_field(basis;fermialg=QuasiFermi(4,0.1),nbandsalg=bands,
                                   mixing=SimpleMixing(),damping=0.5,tol=1e-8,
                                   maxiter=100,callback=identity,seed=UInt64(42))
    @test result.converged
    @test result.μ_e == result.εF.μ_e
    @test result.μ_h == result.εF.μ_h
    @test sum(result.occupation[1][5:end]) ≈ 0.1 atol=1e-10
    @test isfinite(result.energies.Entropy)
    using JLD2
    mktempdir() do dir
        filename = joinpath(dir,"light.jld2")
        save_scfres(filename,result)
        restored = load_scfres(filename)
        @test restored.εF.μ_e == result.εF.μ_e
        @test restored.μ_h == result.μ_h
        @test restored.fermialg.n_eh == 0.1
        @test restored.energies.total ≈ result.energies.total
    end
    @test_throws ErrorException compute_stresses_cart(result)
    @test_throws ErrorException phonon_modes(result)
    @test_throws ErrorException self_consistent_field(basis;fermialg=QuasiFermi(4,0.1))
    algorithm = FixedOccupations(result.occupation)
    fixed = self_consistent_field(basis;fermialg=algorithm,nbandsalg=bands,
                                  mixing=SimpleMixing(),damping=0.5,tol=1e-8,
                                  ρ=result.ρ,ψ=result.ψ,callback=identity,seed=UInt64(42))
    @test fixed.converged
    @test fixed.occupation == algorithm.occupation
    @test fixed.energies.Entropy ≈ result.energies.Entropy
    @test_throws ArgumentError DFTK.compute_occupation(basis,result.eigenvalues,
                                                      FixedOccupations([zeros(12)]))
    @test_throws DimensionMismatch DFTK.compute_occupation(basis,result.eigenvalues,
                                                           FixedOccupations([fill(2.0,4)]))
end

@testitem "Photo occupations: variational L1 forces" tags=[:photo] setup=[TestCases] begin
    using DFTK
    using .TestCases: silicon
    positions = [silicon.positions[1]+[0.02,0.01,0.0],silicon.positions[2]]
    model = model_DFT(silicon.lattice,silicon.atoms,positions;functionals=LDA(),
                      temperature=0.01,smearing=Smearing.FermiDirac(),symmetries=false)
    basis = PlaneWaveBasis(model;Ecut=5,kgrid=[1,1,1])
    kwargs = (;fermialg=QuasiFermi(4,0.1),mixing=SimpleMixing(),damping=0.5,
               nbandsalg=FixedBands(;n_bands_converge=12,n_bands_compute=12),
               tol=1e-10,maxiter=150,callback=identity,seed=UInt64(42))
    result = self_consistent_field(basis;kwargs...)
    @test result.converged
    force = compute_forces_cart(result)[1][1]
    h = 1e-4
    function energy_at(δ)
        shifted = [positions[1]+basis.model.inv_lattice*[δ,0,0],positions[2]]
        b = PlaneWaveBasis(basis;model=Model(model;positions=shifted))
        r = self_consistent_field(b;ρ=result.ρ,ψ=result.ψ,kwargs...)
        @test r.converged
        r.energies.total
    end
    @test force ≈ -(energy_at(h)-energy_at(-h))/(2h) atol=2e-5 rtol=2e-3
end
