@testitem "External potential from Fourier coefficients" #=
    =#    tags=[:dont_test_mpi] begin
    using DFTK

    lattice = [[10 0 0.]; [0 0 0]; [0 0 0]]

    pot(G) = G == 0 ? zero(G) : 1 / abs(G)
    C = 1
    α = 2
    terms = [Kinetic(),
             ExternalFromFourier(G -> pot(G[1])),
             LocalNonlinearity(ρ -> C * ρ^α)]
    model = Model(lattice; n_electrons=1, terms, spin_polarization=:spinless)
    basis = PlaneWaveBasis(model; Ecut=15, kgrid=(1, 1, 1))
    scfres_dm  = direct_minimization(basis; tol=1e-10)
    scfres_scf = self_consistent_field(basis; is_converged=DFTK.ScfConvergenceEnergy(1e-10))
    @test abs(scfres_scf.energies.total - scfres_dm.energies.total) < 1e-6
end

@testitem "Spin-resolved local potential energy" tags=[:minimal, :zeeman] begin
    using DFTK
    using LinearAlgebra

    lattice = 5.0 * Matrix{Float64}(I, 3, 3)
    fft_size = (9, 9, 9)
    potential = zeros(fft_size..., 2)
    potential[:, :, :, 1] .= -0.1
    potential[:, :, :, 2] .= 0.1
    model = Model(lattice; n_electrons=1, spin_polarization=:collinear,
                  terms=[DFTK.ExternalFromValues(potential)], symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=1, kgrid=(1, 1, 1), fft_size=fft_size)
    ρ = zeros(fft_size..., 2)
    ρ[:, :, :, 1] .= 0.8 / model.unit_cell_volume
    ρ[:, :, :, 2] .= 0.2 / model.unit_cell_volume
    result = DFTK.ene_ops(only(basis.terms), basis, nothing, nothing; ρ=ρ)
    @test result.E ≈ -0.06 atol=1e-14
    @test result.E ≈ sum(ρ .* potential) * basis.dvol atol=1e-14
    for (ik, kpt) in enumerate(basis.kpoints)
        @test result.ops[ik].potential == potential[:, :, :, kpt.spin]
    end

    for values in (fill(0.3, fft_size), fill(0.3, fft_size..., 1))
        scalar_model = Model(lattice; n_electrons=1, spin_polarization=:none,
                             terms=[DFTK.ExternalFromValues(values)], symmetries=false)
        scalar_basis = PlaneWaveBasis(scalar_model; Ecut=1, kgrid=(1, 1, 1),
                                     fft_size=fft_size)
        scalar_ρ = fill(inv(scalar_model.unit_cell_volume), fft_size..., 1)
        scalar_result = DFTK.ene_ops(only(scalar_basis.terms), scalar_basis,
                                   nothing, nothing; ρ=scalar_ρ)
        @test scalar_result.E ≈ 0.3 atol=1e-14
    end
end
