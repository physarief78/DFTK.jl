@testitem "RPA: static complete-basis response versus equilibrium χ0" tags=[:rpa] begin
    using DFTK
    using LinearAlgebra
    lattice = [10.0 0 0; 0 0 0; 0 0 0]
    model = Model(lattice; n_electrons=2, terms=[Kinetic()], symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=1, kgrid=[1, 1, 1])
    ham = Hamiltonian(basis)
    F = eigen(Hermitian(Matrix(ham[1])))
    occupation = DFTK.compute_occupation(basis, [F.values])
    data = (; basis, ψ=[F.vectors], eigenvalues=[F.values], occupation=occupation.occupation,
             εF=occupation.εF, ρ=compute_density(basis, [F.vectors], occupation.occupation))
    result = compute_χ0_dynamic(data, [0, 0, 0], [0.0]; η=0, Ecut_response=10,
                                Gs=collect(vec(G_vectors(basis))))
    χreal = compute_χ0(ham)
    reference = DFTK.fft_matrix(basis)*χreal*DFTK.ifft_matrix(basis)
    @test result.χ0[:, :, 1] ≈ reference atol=1e-12 rtol=1e-10
    @test result.χ0[:, :, 1] ≈ result.χ0[:, :, 1]'
    @test maximum(eigvals(Hermitian(result.χ0[:, :, 1]))) ≤ 1e-12
    @test_throws ErrorException compute_χ0_dynamic(data, [0, 0, 0], [0.0];
        η=0, Ecut_response=10, max_matrix_bytes=0)
end

@testitem "RPA: optical head, dense screening and EELS" tags=[:rpa] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    model = model_atomic(TestCases.silicon.lattice, TestCases.silicon.atoms,
                          TestCases.silicon.positions; symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=ExplicitKpoints([[0.13, 0.07, -0.04]]))
    ρ = guess_density(basis)
    F = eigen(Hermitian(Matrix(Hamiltonian(basis; ρ)[1])))
    data = (; basis, ρ, ψ=[F.vectors[:, 1:8]], eigenvalues=[F.values[1:8]],
             occupation=[vcat(fill(2.0, 4), zeros(4))])
    ω = [0.0, 0.1, 0.3, 0.6]
    η = 0.01
    for scissor in (0.0, 0.05)
        response = compute_rpa(data, [0, 0, 0], ω; η, Ecut_response=1,
            Gs=[Vec3(0, 0, 0)], optical_direction=[1.0, 0, 0], scissor, n_valence=4)
        optical = compute_dielectric(data, ω; η, scissor, n_valence=4)
        @test response.εmacroscopic ≈ optical.ε[1, 1, :] rtol=1e-10 atol=1e-12
        @test compute_eels(response).loss ≈ loss_function.(optical.ε[1, 1, :])
    end
    response = compute_χ0_dynamic(data, [0, 0, 0], [0.2]; η, Ecut_response=0.5)
    screened = compute_rpa(response)
    v = Diagonal(vec(DFTK.eval_kernel_fourier(screened.kernel, basis, Vec3(0., 0., 0.)))[
                  DFTK._response_indices(basis, response.Gs)])
    @test screened.εinverse[:, :, 1]*screened.ε[:, :, 1] ≈ I
    @test screened.W[:, :, 1] ≈ (I-v*response.χ0[:, :, 1]) \ Matrix(v)
end

@testitem "RPA: finite-q Sternheimer response" tags=[:rpa] begin
    using DFTK
    using LinearAlgebra
    model = Model([10.0 0 0; 0 0 0; 0 0 0]; n_electrons=2, temperature=0.03,
                  smearing=Smearing.FermiDirac(), terms=[Kinetic(), Hartree()], symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=2, kgrid=[4, 1, 1])
    ham = Hamiltonian(basis; ρ=fill(2/model.unit_cell_volume, basis.fft_size..., 1))
    Fs = [eigen(Hermitian(Matrix(h))) for h in ham.blocks]
    # Sternheimer uses the same number of bands at every k; orbital spheres differ.
    nb = minimum(F -> length(F.values), Fs)
    ψ, ε = [F.vectors[:, 1:nb] for F in Fs], [F.values[1:nb] for F in Fs]
    occupations = DFTK.compute_occupation(basis, ε; tol_n_elec=1e-12)
    ρ = compute_density(basis, ψ, occupations.occupation)
    data = (; basis, ham, ψ, ρ, τ=nothing, occupation_threshold=1e-12, eigenvalues=ε, occupations...)
    derivative = compute_occupation_derivative(basis, ε, data.εF)
    q = Vec3(0.25, 0, 0)
    result = compute_χ0_dynamic(data, q, [0.0]; η=0, Ecut_response=0.5,
                                occupation_derivative=derivative)
    perturbation_G = zeros(ComplexF64, basis.fft_size)
    perturbation_G[1] = 1
    δV = reshape(ifft(basis, perturbation_G), basis.fft_size..., 1)
    direct = apply_χ0(ham, ψ, data.occupation, data.εF, ε, δV;
                      q, tol=1e-10, occupation_threshold=1e-12).δρ
    actual = vec(fft(basis, total_density(direct)))[DFTK._response_indices(basis, result.Gs)]
    head = findfirst(iszero, result.Gs)
    @test actual ≈ result.χ0[:, head, 1] atol=1e-8 rtol=1e-7
    dielectric = DFTK.DielectricAdjoint(data; q)
    action = DFTK.mul_approximate(dielectric, vec(δV); rtol=1e-10).Ax
    action_G = fft(basis, reshape(action, basis.fft_size))
    screened = compute_rpa(result)
    @test screened.εinverse[head, head, 1] ≈ inv(action_G[1]) atol=1e-8 rtol=1e-7
end
