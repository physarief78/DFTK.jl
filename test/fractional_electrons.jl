@testmodule FractionalElectrons begin
    using DFTK
    using PseudoPotentialData

    lattice = [5.39697192863632 2.69848596431816 2.69848596431816;
               0.0              4.67391479368660 1.55797159787754;
               0.0              0.0              4.40660912710674]
    atoms = [ElementPsp(:Al, PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))]
    positions = [zeros(3)]

    function aluminium_model(; interacting=false, n_electrons=3, kwargs...)
        constructor = interacting ? model_DFT : model_atomic
        functionals = interacting ? (; functionals=LDA()) : (;)
        constructor(lattice, atoms, positions; n_electrons, temperature=0.01,
                    smearing=Smearing.Gaussian(), disable_electrostatics_check=true,
                    functionals..., kwargs...)
    end

    function run_aluminium(; kwargs...)
        model = aluminium_model(; kwargs...)
        basis = PlaneWaveBasis(model; Ecut=10, kgrid=[4, 4, 4])
        nbandsalg = FixedBands(; n_bands_converge=8, occupation_threshold=1e-12)
        self_consistent_field(basis; tol=1e-11, maxiter=150, callback=identity,
                              nbandsalg, seed=UInt64(42))
    end

    function metric(group; kwargs...)
        for (key, value) in pairs(kwargs)
            println("METRIC,", group, ",", key, ",", value)
        end
        flush(stdout)
    end
end

@testitem "Fractional F1: preserve integer electron counts" #=
    =# tags=[:fractional] setup=[FractionalElectrons] begin
    using DFTK
    using .FractionalElectrons: aluminium_model, run_aluminium, metric

    for count in (130, 130.0, Float32(130), 130 // 1)
        model = aluminium_model(; n_electrons=count)
        @test model.n_electrons === 130
    end
    @test aluminium_model().n_electrons === 3
    @test aluminium_model(; n_electrons=3.0).n_electrons === 3

    # Compare the two input types on a cheap physical integer-filling system.
    integer = run_aluminium(; n_electrons=3)
    real_integer = run_aluminium(; n_electrons=3.0)
    @test integer.converged
    @test real_integer.converged
    @test integer.energies.total ≈ real_integer.energies.total atol=1e-12 rtol=0
    @test integer.eigenvalues == real_integer.eigenvalues
    @test integer.occupation == real_integer.occupation
    @test integer.ρ == real_integer.ρ

    # Exercise the literal 130-electron inputs in a cheap 1D free-electron model.
    function free_electrons(count)
        lattice = [60.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        model = Model(lattice; n_electrons=count, temperature=0.01,
                      smearing=Smearing.Gaussian(), symmetries=false)
        basis = PlaneWaveBasis(model; Ecut=8, kgrid=[1, 1, 1])
        ρ = fill(130 / model.unit_cell_volume, basis.fft_size..., 1)
        nbandsalg = FixedBands(; n_bands_converge=65, occupation_threshold=1e-12)
        self_consistent_field(basis; ρ, nbandsalg, tol=1e-11,
                              eigensolver=diag_full, callback=identity, seed=UInt64(42))
    end
    free_int = free_electrons(130)
    free_real = free_electrons(130.0)
    @test free_int.converged
    @test free_real.converged
    @test free_int.energies.total ≈ free_real.energies.total atol=1e-12 rtol=0
    @test sum(total_density(free_int.ρ)) * free_int.basis.dvol ≈ 130 atol=1e-10 rtol=0
    # Fill g=0 and the pairs g=±1,...,±32, with two electrons per orbital.
    energy_exact = 2 * (2π / 60)^2 * sum(abs2, 1:32)
    @test free_int.energies.total ≈ energy_exact atol=1e-10 rtol=0
    metric("F1"; energy_int=integer.energies.total,
           energy_real=real_integer.energies.total,
           energy_difference=real_integer.energies.total - integer.energies.total,
           energy_130_int=free_int.energies.total,
           energy_130_real=free_real.energies.total, energy_130_exact=energy_exact,
           energy_130_difference=free_real.energies.total - free_int.energies.total)
end

@testitem "Fractional F2: argument checks" #=
    =# tags=[:fractional, :minimal] setup=[FractionalElectrons] begin
    using DFTK
    using .FractionalElectrons: aluminium_model

    @test_throws r"fractional electron counts require temperature > 0" aluminium_model(
        ; n_electrons=3.3, temperature=0)
    @test_throws r"non-neutral cells" aluminium_model(
        ; n_electrons=3.3, disable_electrostatics_check=false)
    for count in (-1, -0.3)
        @test_throws r"non-negative" aluminium_model(; n_electrons=count)
    end
    for count in (NaN, Inf, -Inf)
        @test_throws r"finite" aluminium_model(; n_electrons=count)
    end
    @test_throws r"finite-temperature smearing" aluminium_model(
        ; n_electrons=3.3, smearing=Smearing.None())
    @test_throws r"both a given" aluminium_model(; n_electrons=3.3, εF=0.1)
    @test_throws r"either n_electrons" aluminium_model(; n_electrons=nothing)

    # An occupation call can override the model's positive temperature.
    basis = PlaneWaveBasis(aluminium_model(; n_electrons=3.3); Ecut=3, kgrid=[1, 1, 1])
    eigenvalues = [collect(1.0:8.0) for _ in basis.kpoints]
    @test_throws r"fractional electron counts require temperature > 0" DFTK.compute_occupation(
        basis, eigenvalues; temperature=0, smearing=Smearing.None())
    @test aluminium_model(; n_electrons=4.0, temperature=0).n_electrons === 4
    @test aluminium_model(; n_electrons=0).n_electrons === 0
    @test aluminium_model(; n_electrons=130.3).n_electrons === 130.3
end

@testitem "Fractional F3: Fermi-level derivative and DOS" #=
    =# tags=[:fractional] setup=[FractionalElectrons] begin
    using DFTK
    using .FractionalElectrons: aluminium_model, run_aluminium, metric

    scfres = run_aluminium(; n_electrons=3.3)
    @test scfres.converged
    basis = scfres.basis
    count = basis.model.n_electrons
    δ = 1e-4
    function occupation_at(n_electrons)
        model = aluminium_model(; n_electrons)
        shifted_basis = PlaneWaveBasis(model; Ecut=basis.Ecut, kgrid=basis.kgrid)
        result = DFTK.compute_occupation(shifted_basis, scfres.eigenvalues;
                                        tol_n_elec=1e-12)
        @test DFTK.weighted_ksum(shifted_basis, sum.(result.occupation)) ≈ n_electrons #=
            =# atol=1e-10 rtol=0
        result.εF
    end
    derivative = (occupation_at(count + δ) - occupation_at(count - δ)) / (2δ)
    dos = sum(compute_dos(scfres))
    @test dos > 0
    @test derivative ≈ inv(dos) rtol=1e-3
    metric("F3"; n_electrons=count, delta=δ, fermi_level=scfres.εF, dos,
           derivative_fd=derivative, inverse_dos=inv(dos),
           relative_error=abs(derivative * dos - 1))
end

@testitem "Fractional F4: interacting doped aluminium" #=
    =# tags=[:fractional] setup=[FractionalElectrons] begin
    using DFTK
    using .FractionalElectrons: run_aluminium, metric

    neutral = run_aluminium(; interacting=true)
    doped = run_aluminium(; interacting=true, n_electrons=3.3)
    @test neutral.converged
    @test doped.converged
    @test isfinite(doped.energies.total)
    count_density = sum(total_density(doped.ρ)) * doped.basis.dvol
    count_occupation = DFTK.weighted_ksum(doped.basis, sum.(doped.occupation))
    @test count_density ≈ 3.3 atol=1e-10 rtol=0
    @test count_occupation ≈ 3.3 atol=1e-10 rtol=0
    @test doped.εF > neutral.εF
    metric("F4"; count_density, count_occupation, energy=doped.energies.total,
           fermi_neutral=neutral.εF, fermi_doped=doped.εF,
           fermi_shift=doped.εF - neutral.εF, iterations=doped.n_iter,
           residual=last(doped.history_Δρ))
end

@testitem "Fractional F5: collinear Hubbard and Zeeman SCF" #=
    =# tags=[:fractional] begin
    using DFTK
    using PseudoPotentialData
    using Unitful
    using UnitfulAtomic

    lattice = 7.9 * [1.0 0.5 0.5; 0.5 1.0 0.5; 0.5 0.5 1.0]
    family = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
    Ni = ElementPsp(:Ni, family; rcut=10.0)
    O = ElementPsp(:O, family; rcut=10.0)
    atoms = [Ni, O, Ni, O]
    positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25],
                 [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]]
    magnetic_moments = [2, 0, -1, 0]
    count = DFTK.n_electrons_from_atoms(atoms) + 0.3
    hubbard = Hubbard(OrbitalManifold(Ni, "3D") => 10u"eV")
    field = MagneticField((0.0, 0.0, 1e-3))
    model = model_DFT(lattice, atoms, positions; functionals=PBE(),
                      extra_terms=[hubbard, field], n_electrons=count,
                      disable_electrostatics_check=true, temperature=0.01,
                      smearing=Smearing.Gaussian(), magnetic_moments)
    basis = PlaneWaveBasis(model; Ecut=15, kgrid=[2, 2, 2])
    nbandsalg = AdaptiveBands(model; occupation_threshold=1e-12)
    scfres = self_consistent_field(basis; tol=1e-10, maxiter=200,
                                  ρ=guess_density(basis, magnetic_moments),
                                  callback=identity, nbandsalg, seed=UInt64(42))
    spin_counts = [sum(selectdim(scfres.ρ, 4, σ)) * basis.dvol for σ = 1:2]
    occupation_count = DFTK.weighted_ksum(basis, sum.(scfres.occupation))
    @test model.spin_polarization == :collinear
    @test scfres.converged
    @test isfinite(scfres.energies.total)
    @test isfinite(scfres.energies.Hubbard)
    @test isfinite(scfres.energies.MagneticField)
    @test sum(spin_counts) ≈ count atol=1e-10 rtol=0
    @test occupation_count ≈ count atol=1e-10 rtol=0
    println("METRIC,F5,n_electrons,", count)
    println("METRIC,F5,n_up,", spin_counts[1])
    println("METRIC,F5,n_down,", spin_counts[2])
    println("METRIC,F5,count_error,", sum(spin_counts) - count)
    println("METRIC,F5,energy,", scfres.energies.total)
    println("METRIC,F5,iterations,", scfres.n_iter)
    println("METRIC,F5,residual,", last(scfres.history_Δρ))
    flush(stdout)
end

@testitem "Fractional F6: SCF serialization round trip" #=
    =# tags=[:fractional, :dont_test_mpi] setup=[FractionalElectrons] begin
    using DFTK
    using JLD2
    using .FractionalElectrons: run_aluminium, metric

    for spin_polarization in (:none, :collinear)
        scfres = run_aluminium(; n_electrons=3.3, spin_polarization)
        @test scfres.converged
        dict = DFTK.scfres_to_dict(scfres)
        @test dict["n_electrons"] === 3.3
        @test dict["spin_polarization"] == spin_polarization
        mktempdir() do tmpdir
            filename = joinpath(tmpdir, "fractional.jld2")
            save_scfres(filename, scfres)
            @test JLD2.load(filename, "n_electrons") === 3.3
            loaded = load_scfres(filename)
            @test loaded.basis.model.n_electrons === 3.3
            @test loaded.basis.model.spin_polarization == spin_polarization
            @test loaded.occupation == scfres.occupation
            @test loaded.eigenvalues == scfres.eigenvalues
            @test loaded.ρ == scfres.ρ
            @test loaded.ψ == scfres.ψ
            @test loaded.εF == scfres.εF
            @test loaded.energies.total ≈ scfres.energies.total atol=1e-12 rtol=0
        end
    end
    metric("F6"; electron_count=3.3, spin_modes="none+collinear")
end

@testitem "Fractional initialization and supercell counts" #=
    =# tags=[:fractional, :minimal] setup=[FractionalElectrons] begin
    using DFTK
    using ForwardDiff
    using .FractionalElectrons: aluminium_model

    for spin_polarization in (:none, :collinear, :full)
        model = aluminium_model(; n_electrons=3.3, spin_polarization, symmetries=false)
        basis = PlaneWaveBasis(model; Ecut=3, kgrid=[2, 1, 1])
        @test model.n_electrons === 3.3
        @test DFTK.default_n_bands(model, 1.0) == ceil(
            Int, 3.3 / (model.n_spin_components * DFTK.filled_occupation(model)))
        for density in (guess_density(basis),
                        @inferred(guess_density(basis, RandomDensity())))
            @test sum(total_density(density)) * basis.dvol ≈ 3.3 atol=1e-12 rtol=0
        end
        supercell = DFTK.cell_to_supercell(basis)
        @test supercell.model.n_electrons ≈ 6.6 atol=1e-14 rtol=0
        @test Model(model; n_electrons=3.5).n_electrons === 3.5
    end
    # Multiplication may make a fractional unit-cell count integral.
    basis = PlaneWaveBasis(aluminium_model(; n_electrons=3.5); Ecut=3, kgrid=[2, 1, 1])
    @test DFTK.cell_to_supercell(basis).model.n_electrons === 7

    model = aluminium_model(; n_electrons=3.3)
    @test Model{Float32}(model).n_electrons === Float32(3.3)
    dual_type = typeof(ForwardDiff.Dual(0.0, 1.0))
    dual_model = Model{dual_type}(model)
    @test dual_model.n_electrons === 3.3
    @test DFTK.construct_value(dual_model).n_electrons === 3.3
end

@testitem "Fractional full-spin atomic SCF" #=
    =# tags=[:fractional, :dont_test_mpi] setup=[FractionalElectrons] begin
    using DFTK
    using .FractionalElectrons: aluminium_model, metric

    model = aluminium_model(; n_electrons=3.3, spin_polarization=:full,
                            symmetries=false, extra_terms=[MagneticField((0, 0, 1e-3))])
    basis = PlaneWaveBasis(model; Ecut=6, kgrid=[2, 2, 2])
    nbandsalg = FixedBands(; n_bands_converge=8, occupation_threshold=1e-12)
    scfres = self_consistent_field(basis; tol=1e-11, maxiter=80, callback=identity,
                                  nbandsalg, seed=UInt64(42))
    count_density = sum(total_density(scfres.ρ)) * basis.dvol
    count_occupation = DFTK.weighted_ksum(basis, sum.(scfres.occupation))
    @test scfres.converged
    @test size(scfres.ρ, 4) == 4
    @test isfinite(scfres.energies.total)
    @test count_density ≈ 3.3 atol=1e-10 rtol=0
    @test count_occupation ≈ 3.3 atol=1e-10 rtol=0
    metric("full_spin"; count_density, count_occupation, energy=scfres.energies.total)
end
