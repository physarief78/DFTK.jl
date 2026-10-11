@testitem "MagneticField: collinear Zeeman splitting" tags=[:minimal, :zeeman,
                                                          :dont_test_mpi] begin
    using DFTK
    using LinearAlgebra

    lattice = 5.0 * Matrix{Float64}(I, 3, 3)
    b = 0.1
    function block_eigenvalues(terms)
        model = Model(lattice; n_electrons=1, terms=terms,
                      spin_polarization=:collinear, symmetries=false)
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))
        ham = energy_hamiltonian(basis, nothing, nothing).ham
        if length(terms) > 1
            @test all(block -> block isa DFTK.DftHamiltonianBlock, ham.blocks)
        end
        [eigvals(Hermitian(Matrix(block))) for block in ham.blocks]
    end
    reference = block_eigenvalues([Kinetic()])
    shifted = block_eigenvalues([Kinetic(), MagneticField([0.0, 0.0, b])])
    @test shifted[1] ≈ reference[1] .- b atol=1e-12
    @test shifted[2] ≈ reference[2] .+ b atol=1e-12
    merged = block_eigenvalues([Kinetic(), ExternalFromReal(r -> 0.3),
                               MagneticField([0.0, 0.0, b])])
    @test merged[1] ≈ reference[1] .+ 0.3 .- b atol=1e-12
    @test merged[2] ≈ reference[2] .+ 0.3 .+ b atol=1e-12
end

@testitem "MagneticField: collinear argument checks and field representations" #=
    =# tags=[:minimal, :zeeman] begin
    using DFTK
    using LinearAlgebra

    lattice = 5.0 * Matrix{Float64}(I, 3, 3)
    fft_size = (9, 9, 9)
    function make_basis(field; spin_polarization=:collinear)
        model = Model(lattice; n_electrons=1, terms=[Kinetic(), MagneticField(field)],
                      spin_polarization=spin_polarization, symmetries=false)
        PlaneWaveBasis(model; Ecut=1, kgrid=(1, 1, 1), fft_size=fft_size)
    end
    @test make_basis([0.0, 0.0, 0.1]) isa PlaneWaveBasis
    @test make_basis((0.0, 0.0, 0.1)) isa PlaneWaveBasis
    @test make_basis(r -> [0.0, 0.0, 0.1 * sin(r[1])]) isa PlaneWaveBasis
    values = zeros(fft_size..., 3)
    values[:, :, :, 3] .= 0.1
    @test make_basis(values).terms[2].field == values
    values[2, 3, 4, 1] = 1e-14
    @test_throws ArgumentError make_basis(values)
    @test_throws ArgumentError make_basis([0.1, 0.0, 0.0])
    @test_throws ArgumentError make_basis(r -> [0.0, 0.1, 0.0])
    for spin in (:none, :spinless)
        @test_throws ArgumentError make_basis([0.0, 0.0, 0.1]; spin_polarization=spin)
    end
    @test_throws DimensionMismatch make_basis([0.0, 0.1])
    @test_throws DimensionMismatch make_basis(r -> [0.0, 0.1])
    @test_throws DimensionMismatch make_basis(zeros(fft_size..., 2))
    @test_throws ArgumentError make_basis("unsupported")
end

@testitem "MagneticField: ForwardDiff field conversion" tags=[:minimal, :zeeman] begin
    using DFTK
    using ForwardDiff
    using LinearAlgebra

    lattice = 5.0 * Matrix{Float64}(I, 3, 3)
    fft_size = (9, 9, 9)
    representations = (
        b -> [zero(b), zero(b), b],
        b -> (zero(b), zero(b), b),
        b -> (r -> [zero(b), zero(b), b]),
        b -> cat(zeros(typeof(b), fft_size), zeros(typeof(b), fft_size),
                 fill(b, fft_size); dims=4),
    )
    for representation in representations
        function zeeman_energy(b; dual_basis=true)
            typed_lattice = dual_basis ? Matrix{typeof(b)}(lattice) : lattice
            model = Model(typed_lattice; n_electrons=1,
                          terms=[MagneticField(representation(b))],
                          spin_polarization=:collinear, symmetries=false)
            basis = PlaneWaveBasis(model; Ecut=1, kgrid=(1, 1, 1), fft_size=fft_size)
            ρ = zeros(eltype(basis), fft_size..., 2)
            ρ[:, :, :, 1] .= inv(model.unit_cell_volume)
            DFTK.ene_ops(only(basis.terms), basis, nothing, nothing; ρ=ρ).E
        end
        @test zeeman_energy(0.1) ≈ -0.1 atol=1e-14
        @test ForwardDiff.derivative(zeeman_energy, 0.1) ≈ -1 atol=1e-13
        @test zeeman_energy(ForwardDiff.Dual(0.1, 1.0); dual_basis=false) ≈ -0.1
    end
end

@testitem "MagneticField: collinear and full LDA agree" tags=[:zeeman] begin
    using DFTK
    using LinearAlgebra
    using PseudoPotentialData

    lattice = 5.42352 / 2 * [-1.0 1 1; 1 -1 1; 1 1 -1]
    Fe = ElementPsp(:Fe, PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth"))
    function solve(spin)
        moments = spin == :full ? [[0.0, 0.0, 4.0]] : [4.0]
        model = model_DFT(lattice, [Fe], [zeros(3)]; functionals=LDA(),
                          temperature=0.02, smearing=Smearing.Gaussian(),
                          spin_polarization=spin, magnetic_moments=moments,
                          symmetries=false, extra_terms=[MagneticField([0.0, 0.0, 0.01])])
        basis = PlaneWaveBasis(model; Ecut=8, kgrid=(2, 2, 2))
        self_consistent_field(basis; ρ=guess_density(basis, moments), tol=1e-11,
                              maxiter=150, seed=1234, callback=identity,
                              eigensolver=diag_full)
    end
    collinear = solve(:collinear)
    full = solve(:full)
    @test collinear.converged && full.converged
    @test abs(collinear.energies.total - full.energies.total) < 1e-8
    mc = sum(spin_density(collinear.ρ)) * collinear.basis.dvol
    mf = sum(full.ρ[:, :, :, 4]) * full.basis.dvol
    @info "Zeeman mode comparison" ΔE=(
        collinear.energies.total-full.energies.total) ΔM=(mc-mf)
    @test abs(mc - mf) < 1e-7
    @test maximum(abs, full.ρ[:, :, :, 2:3]) < 1e-10
end

@testitem "MagneticField: uniform fields preserve spatial symmetry" tags=[:zeeman] begin
    using DFTK
    using PseudoPotentialData

    lattice = 5.42352 / 2 * [-1.0 1 1; 1 -1 1; 1 1 -1]
    Fe = ElementPsp(:Fe, PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth"))
    function make_model(extra_terms; symmetries=true)
        model_DFT(lattice, [Fe], [zeros(3)]; functionals=LDA(),
                  temperature=0.02, smearing=Smearing.Gaussian(),
                  spin_polarization=:collinear, magnetic_moments=[4.0],
                  symmetries=symmetries, extra_terms=extra_terms)
    end
    reference = make_model([])
    uniform = make_model([MagneticField([0.0, 0.0, 0.01])])
    @test uniform.symmetries == reference.symmetries
    @test length(uniform.symmetries) > 1
    @test !DFTK.breaks_symmetries(MagneticField((0.0, 0.0, 0.01)))
    @test DFTK.breaks_time_reversal_symmetry(MagneticField([0.0, 0.0, 0.01]))
    varying = make_model([MagneticField(r -> [0.0, 0.0, 0.01 * cos(r[1])])])
    @test length(varying.symmetries) == 1
    @test isone(only(varying.symmetries))
    function solve(model)
        basis = PlaneWaveBasis(model; Ecut=8, kgrid=(2, 2, 2))
        self_consistent_field(basis; ρ=guess_density(basis, [4.0]), tol=1e-11,
                              maxiter=150, seed=1234, callback=identity,
                              eigensolver=diag_full)
    end
    with_symmetries = solve(uniform)
    without = solve(make_model([MagneticField([0.0, 0.0, 0.01])]; symmetries=false))
    @test with_symmetries.converged && without.converged
    @test abs(with_symmetries.energies.total - without.energies.total) < 1e-8
    moment(result) = sum(spin_density(result.ρ)) * result.basis.dvol
    @test abs(moment(with_symmetries) - moment(without)) < 1e-8
    @info "Zeeman symmetry comparison" nsym=length(uniform.symmetries) ΔE=(
        with_symmetries.energies.total-without.energies.total)
end

@testitem "MagneticField: Pauli susceptibility identity" tags=[:zeeman] begin
    using DFTK
    using PseudoPotentialData

    lattice = 7.65 / 2 * [0.0 1 1; 1 0 1; 1 1 0]
    Al = ElementPsp(:Al, PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth"))
    function solve(b)
        model = model_atomic(lattice, [Al], [zeros(3)]; spin_polarization=:collinear,
                             temperature=0.01, smearing=Smearing.Gaussian(),
                             magnetic_moments=[0.0],
                             extra_terms=[MagneticField([0.0, 0.0, b])])
        basis = PlaneWaveBasis(model; Ecut=10, kgrid=(4, 4, 4))
        self_consistent_field(basis; tol=1e-11, seed=1234, callback=identity,
                              eigensolver=diag_full)
    end
    moment(result) = sum(spin_density(result.ρ)) * result.basis.dvol
    δb = 1e-5
    plus, minus, zero_field = solve(δb), solve(-δb), solve(0.0)
    @test plus.converged && minus.converged && zero_field.converged
    dMdb = (moment(plus) - moment(minus)) / (2δb)
    dos = sum(compute_dos(zero_field))
    @info "Pauli identity" dMdb dos relative_error=abs(dMdb-dos)/abs(dos)
    @test dMdb > 0
    @test dMdb ≈ dos rtol=1e-4
end
