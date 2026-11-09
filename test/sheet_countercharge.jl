@testitem "Sheet countercharge: neutrality, Poisson and ionic forces" tags=[:photo] begin
    using DFTK
    using LinearAlgebra
    lattice = diagm([8.0, 8.0, 20.0])
    sheet = SheetCounterCharge(0.2; center=0.8, width=0.8)
    model = Model(lattice, [ElementCoulomb(:H)], [[0.4, 0.5, 0.4]];
                  n_electrons=1.2, temperature=0.01, terms=[Kinetic(), sheet], symmetries=false,
                  disable_electrostatics_check=true)
    basis = PlaneWaveBasis(model; Ecut=1, kgrid=[1, 1, 1], fft_size=(9, 9, 64))
    term = basis.terms[2]
    @test sum(term.charge_density)*basis.dvol ≈ 0.2
    VG = fft(basis, term.potential_values)
    ρG = fft(basis, term.charge_density)
    laplacian = [-sum(abs2, G) for G in G_vectors_cart(basis)] .* VG
    source = 4π .* ρG
    source[1] = 0
    @test laplacian ≈ source rtol=1e-12 atol=1e-12
    ρ = guess_density(basis)
    E = DFTK.ene_ops(term, basis, nothing, nothing; ρ).E
    @test isfinite(E)
    @test_throws ErrorException DFTK.compute_dynmat(term, basis, nothing, nothing)
    h = 1e-5
    function energy_at(z)
        m = Model(model; positions=[[0.4, 0.5, z]])
        b = PlaneWaveBasis(basis; model=m)
        DFTK.ene_ops(b.terms[2], b, nothing, nothing; ρ).E
    end
    force = DFTK.compute_forces(term, basis, nothing, nothing)[1][3]
    @test force ≈ -(energy_at(0.4+h)-energy_at(0.4-h))/(2h) rtol=1e-8
    @test_throws ErrorException PlaneWaveBasis(Model(model; n_electrons=1.1); Ecut=1)
end
