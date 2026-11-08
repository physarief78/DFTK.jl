@testitem "Noncollinear algebra and component API" tags=[:noncollinear, :minimal,
                                                            :dont_test_mpi] begin
    using DFTK
    using ForwardDiff
    using LinearAlgebra
    using StaticArrays
    using Test

    for C in (1, 2, 3)
        basis = hermitian_basis(Val(C))
        gram = [real(tr(left' * right)) for left in basis, right in basis]
        @test gram ≈ Matrix{Float64}(I, C^2, C^2) atol=5e-15

        fields = SVector{C^2,Float64}(ntuple(i -> sin(i), C^2))
        matrix = density_matrix(fields)
        @test pauli_components(matrix) ≈ fields atol=5e-15

        potential = SVector{C^2,Float64}(ntuple(i -> cos(i), C^2))
        duality = real(tr(DFTK.potential_matrix(potential) * matrix))
        @test duality ≈ dot(potential, fields) atol=5e-15
    end

    fields = SVector(1.2, 0.1, -0.2, 0.3)
    @test @inferred(pauli_components(density_matrix(fields))) ≈ fields
    derivative = ForwardDiff.derivative(0.4) do x
        transformed = pauli_components(density_matrix(SVector(x, 0.1, -0.2, 0.3)))
        sum(transformed)
    end
    @test derivative ≈ 1

    lattice = 5.0 .* Matrix{Float64}(I, 3, 3)
    model_scalar = Model(lattice; n_electrons=1, terms=[Kinetic()],
                         spin_polarization=:spinless, symmetries=false)
    model_full = Model(lattice; n_electrons=1, terms=[Kinetic()],
                       spin_polarization=:full, symmetries=false)
    @test n_components(model_scalar) == 1
    @test n_density_components(model_scalar) == 1
    @test n_components(model_full) == 2
    @test n_density_components(model_full) == 4
    @test DFTK.filled_occupation(model_full) == 1
end

@testitem "Noncollinear orbitals, density, and Hamiltonian" tags=[:noncollinear,
                                                                    :minimal,
                                                                    :dont_test_mpi] begin
    using DFTK
    using LinearAlgebra
    using Random
    using StaticArrays
    using Test

    Random.seed!(1234)
    lattice = 5.0 .* Matrix{Float64}(I, 3, 3)
    field = [0.2, -0.1, 0.05]
    model = Model(lattice; n_electrons=1,
                  terms=[Kinetic(), MagneticField(field)],
                  spin_polarization=:full, symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=1, kgrid=(1, 1, 1), fft_size=(9, 9, 9))
    kpoint = only(basis.kpoints)
    n_G = length(G_vectors(basis, kpoint))

    ψ = DFTK.random_orbitals(basis, kpoint, 3)
    @test size(ψ) == (2, n_G, 3)
    @test as_matrix(ψ)' * as_matrix(ψ) ≈ I atol=2e-14
    @test component_view(ψ, 1) == each_component(ψ)[1]
    ψspinors = spinors(ψ)
    @test size(ψspinors) == (n_G, 3)
    @test ψspinors[1, 1] == SVector(ψ[1, 1, 1], ψ[2, 1, 1])

    ψreal = ntuple(_ -> zeros(ComplexF64, basis.fft_size), 2)
    DFTK.ifft_components!(ψreal, basis, kpoint, ψ, 1)
    ψroundtrip = zero(ψ)
    DFTK.fft_components!(ψroundtrip, 1, basis, kpoint, ψreal)
    @test ψroundtrip[:, :, 1] ≈ ψ[:, :, 1] atol=2e-14

    ham = Hamiltonian(basis)
    Hk = only(ham.blocks)
    Hmatrix = Matrix(Hk)
    Hψ = Hk * as_matrix(ψ)
    @test Hψ ≈ Hmatrix * as_matrix(ψ) atol=2e-13
    ψvector = copy(@view as_matrix(ψ)[:, 1])
    @test Hk * ψvector ≈ Hmatrix * ψvector atol=2e-13
    @test Hmatrix ≈ Hmatrix' atol=2e-14

    φ = DFTK.random_orbitals(basis, kpoint, 3)
    @test dot(as_matrix(φ), Hk * as_matrix(ψ)) ≈
          dot(Hk * as_matrix(φ), as_matrix(ψ)) atol=2e-13

    eigenvalues = eigen(Hermitian(Hmatrix)).values
    @test eigenvalues[1:2] ≈ [-norm(field), norm(field)] atol=2e-14
    eigres = diagonalize_all_kblocks(diag_full, ham, 4; interpolate_kpoints=false)
    @test size(only(eigres.X)) == (2, n_G, 4)
    @test only(eigres.λ)[1:2] ≈ [-norm(field), norm(field)] atol=2e-14

    # A normalized Γ-only |+x⟩ spinor has n=mx=1/Ω and my=mz=0 everywhere.
    ψx = zeros(ComplexF64, 2, n_G, 1)
    iGzero = only(findall(iszero, G_vectors(basis, kpoint)))
    ψx[:, iGzero, 1] .= inv(sqrt(2))
    ρ = compute_density(basis, [ψx], [ones(1)])
    reference_density = inv(model.unit_cell_volume)
    @test size(ρ) == (basis.fft_size..., 4)
    @test ρ[:, :, :, 1] ≈ fill(reference_density, basis.fft_size) atol=2e-15
    @test ρ[:, :, :, 2] ≈ fill(reference_density, basis.fft_size) atol=2e-15
    @test maximum(abs, ρ[:, :, :, 3:4]) < 2e-15
    @test sum(total_density(ρ)) * basis.dvol ≈ 1 atol=2e-14
    @test spin_projections(ψx) ≈ [1.0 0.0 0.0] atol=2e-15

    ρroundtrip = zero(ρ)
    from_density_matrices!(ρroundtrip, density_matrices(ρ))
    @test ρroundtrip ≈ ρ atol=2e-15

    packed = DFTK.blockify_ψ(basis, [ψ])
    unpacked = DFTK.unblockify_ψ(packed.ψ, packed.n_G_vectors)
    @test only(unpacked) == ψ

    ρrandom = @inferred guess_density(basis, RandomDensity())
    mrandom = magnetization(basis, ρrandom)
    mnorm = dropdims(sqrt.(sum(abs2, mrandom; dims=4)); dims=4)
    @test minimum(total_density(ρrandom) .- mnorm) ≥ -20eps(Float64)
    @test sum(total_density(ρrandom)) * basis.dvol ≈ model.n_electrons
end

@testitem "Noncollinear LDA, mixing, and SCF" tags=[:noncollinear, :minimal,
                                                       :dont_test_mpi] begin
    using DFTK
    using LinearAlgebra
    using Test

    lattice = 5.0 .* Matrix{Float64}(I, 3, 3)
    xc = Xc([:lda_x])
    model_full = Model(lattice; n_electrons=1, terms=[xc],
                       spin_polarization=:full, symmetries=false)
    model_collinear = Model(lattice; n_electrons=1, terms=[xc],
                            spin_polarization=:collinear, symmetries=false)
    basis_full = PlaneWaveBasis(model_full; Ecut=1, kgrid=(1, 1, 1),
                                fft_size=(9, 9, 9))
    basis_collinear = PlaneWaveBasis(model_collinear; Ecut=1, kgrid=(1, 1, 1),
                                     fft_size=(9, 9, 9))

    n = fill(0.02, basis_full.fft_size)
    m = zeros(basis_full.fft_size..., 3)
    m[:, :, :, 1] .= 0.004
    m[:, :, :, 2] .= -0.003
    m[:, :, :, 3] .= 0.005
    ρ = ρ_from_total_and_magnetization(n, m)
    δρ = 1e-3 .* reshape(sin.(1:length(ρ)), size(ρ))
    ε = 1e-5

    energy_ham = energy_hamiltonian(basis_full, nothing, nothing; ρ)
    potential = DFTK.total_local_potential(energy_ham.ham)
    Eplus = DFTK.energy(basis_full, nothing, nothing; ρ=ρ + ε * δρ).energies.total
    Eminus = DFTK.energy(basis_full, nothing, nothing; ρ=ρ - ε * δρ).energies.total
    finite_difference = (Eplus - Eminus) / (2ε)
    predicted = sum(potential .* δρ) * basis_full.dvol
    @test finite_difference ≈ predicted rtol=2e-6 atol=2e-10

    # A full-spin density aligned with z is exactly the existing collinear problem.
    mz = fill(0.005, basis_full.fft_size)
    m_z = cat(zero(mz), zero(mz), mz; dims=Val(4))
    ρ_full_z = ρ_from_total_and_magnetization(n, m_z)
    ρ_collinear = ρ_from_total_and_spin(n, mz)
    result_full_z = energy_hamiltonian(basis_full, nothing, nothing; ρ=ρ_full_z)
    result_collinear = energy_hamiltonian(basis_collinear, nothing, nothing; ρ=ρ_collinear)
    Vfull = DFTK.total_local_potential(result_full_z.ham)
    Vcollinear = DFTK.total_local_potential(result_collinear.ham)
    @test result_full_z.energies.total ≈ result_collinear.energies.total atol=2e-14
    @test Vfull[:, :, :, 1] + Vfull[:, :, :, 4] ≈ Vcollinear[:, :, :, 1] atol=5e-15
    @test Vfull[:, :, :, 1] - Vfull[:, :, :, 4] ≈ Vcollinear[:, :, :, 2] atol=5e-15
    @test maximum(abs, Vfull[:, :, :, 2:3]) == 0

    # Rotating z to x rotates the XC magnetic field and leaves the energy invariant.
    m_x = cat(mz, zero(mz), zero(mz); dims=Val(4))
    ρ_full_x = ρ_from_total_and_magnetization(n, m_x)
    result_full_x = energy_hamiltonian(basis_full, nothing, nothing; ρ=ρ_full_x)
    Vfull_x = DFTK.total_local_potential(result_full_x.ham)
    @test result_full_x.energies.total ≈ result_full_z.energies.total atol=2e-14
    @test Vfull_x[:, :, :, 2] ≈ Vfull[:, :, :, 4] atol=5e-15
    @test maximum(abs, Vfull_x[:, :, :, 3:4]) == 0

    δF = reshape(cos.(1:(prod(basis_full.fft_size) * 4)), basis_full.fft_size..., 4)
    kerker = DFTK.mix_density(KerkerMixing(), basis_full, δF)
    @test kerker[:, :, :, 2:4] == δF[:, :, :, 2:4]
    ldos = DFTK.mix_density(LdosMixing(), basis_full, copy(δF); ρin=ρ_full_z)
    @test ldos ≈ kerker atol=2e-14

    # End-to-end SCF for a uniform electron gas polarized by a transverse field.
    model_scf = Model(lattice; n_electrons=1,
                      terms=[Kinetic(), Hartree(), Xc([:lda_x]),
                             MagneticField([0.05, 0.0, 0.0])],
                      spin_polarization=:full, symmetries=false)
    basis_scf = PlaneWaveBasis(model_scf; Ecut=1, kgrid=(1, 1, 1),
                               fft_size=(9, 9, 9))
    n0 = fill(inv(model_scf.unit_cell_volume), basis_scf.fft_size)
    m0 = cat(n0, zero(n0), zero(n0); dims=Val(4))
    ρ0 = ρ_from_total_and_magnetization(n0, m0)
    scfres = self_consistent_field(
        basis_scf; ρ=ρ0, tol=1e-9, maxiter=4, mixing=SimpleMixing(), damping=0.7,
        solver=ScfDampingSolver(), eigensolver=diag_full,
        nbandsalg=FixedBands(; n_bands_converge=1, n_bands_compute=4),
        callback=_ -> nothing,
    )
    integrated_m = [sum(scfres.ρ[:, :, :, i]) * basis_scf.dvol for i in 2:4]
    @test scfres.converged
    @test scfres.n_iter == 1
    @test size(scfres.ρ) == (basis_scf.fft_size..., 4)
    @test size(only(scfres.ψ)) == (2, length(G_vectors(basis_scf, only(basis_scf.kpoints))), 4)
    @test sum(total_density(scfres.ρ)) * basis_scf.dvol ≈ 1 atol=2e-14
    @test integrated_m ≈ [1.0, 0.0, 0.0] atol=2e-14
    @test isfinite(scfres.energies.total)
    @test_throws ErrorException compute_forces(scfres)

    model_gga = Model(lattice; n_electrons=1, terms=[PBE()],
                      spin_polarization=:full, symmetries=false)
    @test_throws ArgumentError PlaneWaveBasis(model_gga; Ecut=1, kgrid=(1, 1, 1),
                                              fft_size=(9, 9, 9))
end

@testitem "Noncollinear serialization" tags=[:noncollinear, :serialisation,
                                               :dont_test_mpi] begin
    using DFTK
    using JLD2
    using LinearAlgebra
    using Test
    using WriteVTK

    lattice = 5.0 .* Matrix{Float64}(I, 3, 3)
    model = Model(lattice; n_electrons=1,
                  terms=[Kinetic(), MagneticField([0.05, 0.0, 0.0])],
                  spin_polarization=:full, symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=1, kgrid=(1, 1, 1), fft_size=(9, 9, 9))
    n0 = fill(inv(model.unit_cell_volume), basis.fft_size)
    ρ0 = ρ_from_total_and_magnetization(n0,
                                        cat(n0, zero(n0), zero(n0); dims=Val(4)))
    scfres = self_consistent_field(
        basis; ρ=ρ0, tol=1e-9, maxiter=3, mixing=SimpleMixing(),
        solver=ScfDampingSolver(), eigensolver=diag_full,
        nbandsalg=FixedBands(; n_bands_converge=1, n_bands_compute=4),
        callback=_ -> nothing,
    )

    dictionary = DFTK.scfres_to_dict(scfres; save_ψ=true, save_ρ=true)
    @test size(dictionary["ψ"])[1:3] == (2, length(G_vectors(basis, only(basis.kpoints))), 4)
    @test size(dictionary["ρ"]) == (basis.fft_size..., 4)

    mktempdir() do directory
        jld2file = joinpath(directory, "noncollinear.jld2")
        save_scfres(jld2file, scfres; save_ψ=true, save_ρ=true)
        loaded = load_scfres(jld2file, basis; skip_hamiltonian=true)
        @test size(only(loaded.ψ)) == size(only(scfres.ψ))
        @test only(loaded.ψ) ≈ only(scfres.ψ)
        @test loaded.ρ ≈ scfres.ρ

        vtkfile = joinpath(directory, "noncollinear.vts")
        save_scfres(vtkfile, scfres; save_ψ=true)
        @test isfile(vtkfile)
    end
end

@testitem "GPU noncollinear density and matrix potential" tags=[:gpu, :noncollinear,
                                                                 :dont_test_mpi] begin
    using AMDGPU
    using CUDA
    using DFTK
    using LinearAlgebra
    using Test

    lattice = 5.0 .* Matrix{Float64}(I, 3, 3)
    model = Model(lattice; n_electrons=1,
                  terms=[Kinetic(), MagneticField([0.2, -0.1, 0.05])],
                  spin_polarization=:full, symmetries=false)
    basis_cpu = PlaneWaveBasis(model; Ecut=1, kgrid=(1, 1, 1), fft_size=(9, 9, 9))
    ψ_cpu = DFTK.random_orbitals(basis_cpu, only(basis_cpu.kpoints), 3)
    occupation = [ones(3)]
    ρ_cpu = compute_density(basis_cpu, [ψ_cpu], occupation)
    Hψ_cpu = only(Hamiltonian(basis_cpu).blocks) * as_matrix(ψ_cpu)
    # Reductions over the orbital components are a different device code path than the
    # density and the Hamiltonian application, since they need contiguous per-component
    # arrays. Note that on a GPU basis the occupations are expected to live on the device.
    kinetic_energies_cpu = basis_cpu.terms[1].kinetic_energies[1]
    band_energies_cpu = DFTK._kinetic_band_energies(ψ_cpu, kinetic_energies_cpu)
    projections_cpu = spin_projections(ψ_cpu)
    energies_cpu = energy_hamiltonian(basis_cpu, [ψ_cpu], occupation; ρ=ρ_cpu).energies

    if CUDA.has_cuda() && CUDA.has_cuda_gpu()
        basis_cuda = PlaneWaveBasis(model; Ecut=1, kgrid=(1, 1, 1), fft_size=(9, 9, 9),
                                    architecture=DFTK.GPU(CuArray))
        ψ_cuda = CuArray(ψ_cpu)
        occupation_cuda = [CuArray(occk) for occk in occupation]
        ρ_cuda = compute_density(basis_cuda, [ψ_cuda], occupation_cuda)
        Hψ_cuda = only(Hamiltonian(basis_cuda).blocks) * as_matrix(ψ_cuda)
        kinetic_energies_cuda = basis_cuda.terms[1].kinetic_energies[1]
        energies_cuda = energy_hamiltonian(basis_cuda, [ψ_cuda], occupation_cuda;
                                           ρ=ρ_cuda).energies
        @test Array(ρ_cuda) ≈ ρ_cpu atol=2e-12
        @test Array(Hψ_cuda) ≈ Hψ_cpu atol=2e-12
        @test Array(DFTK._kinetic_band_energies(ψ_cuda, kinetic_energies_cuda)) ≈
              band_energies_cpu atol=2e-12
        @test Array(spin_projections(ψ_cuda)) ≈ projections_cpu atol=2e-12
        @test energies_cuda.total ≈ energies_cpu.total atol=2e-12
        @test energies_cuda["Kinetic"] ≈ energies_cpu["Kinetic"] atol=2e-12
    end

    if AMDGPU.has_rocm_gpu()
        basis_rocm = PlaneWaveBasis(model; Ecut=1, kgrid=(1, 1, 1), fft_size=(9, 9, 9),
                                    architecture=DFTK.GPU(ROCArray))
        ψ_rocm = ROCArray(ψ_cpu)
        occupation_rocm = [ROCArray(occk) for occk in occupation]
        ρ_rocm = compute_density(basis_rocm, [ψ_rocm], occupation_rocm)
        Hψ_rocm = only(Hamiltonian(basis_rocm).blocks) * as_matrix(ψ_rocm)
        kinetic_energies_rocm = basis_rocm.terms[1].kinetic_energies[1]
        energies_rocm = energy_hamiltonian(basis_rocm, [ψ_rocm], occupation_rocm;
                                           ρ=ρ_rocm).energies
        @test Array(ρ_rocm) ≈ ρ_cpu atol=2e-12
        @test Array(Hψ_rocm) ≈ Hψ_cpu atol=2e-12
        @test Array(DFTK._kinetic_band_energies(ψ_rocm, kinetic_energies_rocm)) ≈
              band_energies_cpu atol=2e-12
        @test Array(spin_projections(ψ_rocm)) ≈ projections_cpu atol=2e-12
        @test energies_rocm.total ≈ energies_cpu.total atol=2e-12
        @test energies_rocm["Kinetic"] ≈ energies_cpu["Kinetic"] atol=2e-12
    end
end
