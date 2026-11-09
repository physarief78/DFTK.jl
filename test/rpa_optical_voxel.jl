@testitem "RPA screened cell: dielectric head, local fields and rotation" tags=[:rpa] begin
    using DFTK, FastGaussQuadrature, LinearAlgebra
    model = Model(8.0Matrix(I, 3, 3); n_electrons=2, terms=[Kinetic()], symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=0.2, kgrid=[4, 4, 4], fft_size=(5, 5, 5))
    kernel = BareCoulomb()
    ε = 4.0
    A = ComplexF64.((1-ε)/(4π)*Matrix(I, 3, 3))
    empty_expansion = (; A=A, B=zeros(ComplexF64, 3, 0), C=zeros(ComplexF64, 0, 0),
        K=zeros(ComplexF64, 1, 1), Gs=[Vec3(0, 0, 0)], body=Int[])
    integrate = DFTK._integrate_screening_voxel
    result = integrate(empty_expansion, kernel, basis, 12)
    bare = DFTK.eval_kernel_fourier(VoxelAverage(kernel), basis, Vec3(0.0, 0, 0))[1]
    @test result.W[1, 1] ≈ bare/ε rtol=1e-12
    @test result.dielectric_tensor ≈ ε*I
    @test result.compressibility == 0
    single = merge(empty_expansion, (; A=ComplexF32.(A), B=zeros(ComplexF32, 3, 0),
        C=zeros(ComplexF32, 0, 0), K=zeros(ComplexF32, 1, 1)))
    single_result = integrate(single, kernel, basis, 12)
    @test eltype(single_result.W) == ComplexF32
    @test single_result.W[1, 1] ≈ bare/ε rtol=50eps(Float32)
    # An isotropic macroscopic tensor with a nonzero microscopic wing.
    G = Vec3(1, 0, 0)
    v = DFTK.eval_kernel_fourier(kernel, model.recip_lattice*G)
    C = fill(-0.1+0im, 1, 1)
    S = v/(1-v*C[1, 1])
    B = reshape(ComplexF64[0.02im, 0.03im, 0.01im], 3, 1)
    expanded = (; A=A-B*S*B', B=B, C=C, K=zeros(ComplexF64, 2, 2),
                  Gs=[Vec3(0, 0, 0), G], body=[2])
    local_fields = integrate(expanded, kernel, basis, 12)
    @test local_fields.W[1, 1] ≈ bare/ε rtol=1e-12
    @test local_fields.W[1, 2] == 0
    @test local_fields.W[2, 2] ≈ S+S^2*4π/(3ε)*sum(abs2, B) rtol=1e-12
    @test local_fields.W ≈ local_fields.W'
    tensor = diagm([2.0, 4.0, 7.0])
    anisotropic = merge(empty_expansion, (; A=ComplexF64.((I-tensor)/(4π))))
    original = integrate(anisotropic, kernel, basis, 16)
    θ = 0.37
    rotation = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1.0]
    rotated_model = Model(rotation*model.lattice; n_electrons=2,
                            terms=[Kinetic()], symmetries=false)
    rotated_basis = PlaneWaveBasis(basis; model=rotated_model)
    rotated = merge(empty_expansion, (; A=ComplexF64.((I-rotation*tensor*rotation')/(4π))))
    @test integrate(rotated, kernel, rotated_basis, 16).W ≈ original.W rtol=1e-12
    @test_throws ArgumentError integrate(empty_expansion, kernel, basis, 11)
end

@testitem "RPA screened cell: slab and carrier limits" tags=[:rpa] begin
    using DFTK, FastGaussQuadrature, LinearAlgebra
    model = Model(diagm([8.0, 9.0, 20.0]); n_electrons=2, terms=[Kinetic()], symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=0.2, kgrid=[4, 6, 1], fft_size=(5, 5, 7))
    expansion = (; A=zeros(ComplexF64, 3, 3), B=zeros(ComplexF64, 3, 0), C=zeros(ComplexF64, 0, 0),
        K=zeros(ComplexF64, 1, 1), Gs=[Vec3(0, 0, 0)], body=Int[])
    kernel = SlabTruncatedCoulomb(basis)
    integrate = DFTK._integrate_screening_voxel
    slab = integrate(expansion, kernel, basis, 16)
    bare = DFTK.eval_kernel_fourier(VoxelAverage(kernel; n_quadrature_points=16),
                                    basis, Vec3(0.0, 0, 0))[1]
    @test slab.W[1, 1] ≈ bare rtol=1e-12
    @test isnothing(slab.dielectric_tensor)
    κ = 1e3
    carriers = merge(expansion, (; K=fill(-κ^2/(4π)+0im, 1, 1)))
    screened = integrate(carriers, BareCoulomb(), basis, 16)
    @test real(screened.W[1, 1]) ≈ 4π/κ^2 rtol=1e-7
    @test real(integrate(carriers, kernel, basis, 16).W[1, 1]) ≈ 4π/κ^2 rtol=1e-7
    @test screened.compressibility ≈ κ^2/(4π)
end

@testitem "RPA screened cell: optical response and carrier plumbing" tags=[:rpa] #=
    =# setup=[TestCases] begin
    using DFTK, FastGaussQuadrature, LinearAlgebra
    model = model_atomic(TestCases.silicon.lattice, TestCases.silicon.atoms,
                          TestCases.silicon.positions; symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=3, kgrid=[2, 2, 2])
    ρ = guess_density(basis)
    ham = Hamiltonian(basis; ρ=ρ)
    Fs = [eigen(Hermitian(Matrix(h))) for h in ham.blocks]
    data = (; basis=basis, ρ=ρ, ψ=[F.vectors[:, 1:8] for F in Fs],
              eigenvalues=[F.values[1:8] for F in Fs],
              occupation=[vcat(fill(2.0, 4), zeros(4)) for F in Fs])
    Gs = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(-1, 0, 0)]
    cell = compute_rpa_screening_voxel(data; Gs=Gs, Ecut_response=1, scissor=0.05, n_valence=4)
    for α=1:3
        direction = Float64.(1:3 .== α)
        response = compute_rpa(data, [0, 0, 0], [0.0]; η=0, Ecut_response=1, Gs=Gs,
            optical_direction=direction, scissor=0.05, n_valence=4)
        @test cell.dielectric_tensor[α, α] ≈ real(only(response.εmacroscopic)) rtol=1e-10
    end
    @test cell.W[:, :, 1] ≈ cell.W[:, :, 1]'
    @test minimum(eigvals(Hermitian(cell.W[:, :, 1]))) > 0
    @test_throws ArgumentError compute_eels(cell)
    degenerate_energies = deepcopy(data.eigenvalues)
    degenerate_energies[1][5] = degenerate_energies[1][4]
    @test_throws ErrorException compute_rpa_screening_voxel(
        merge(data, (; eigenvalues=degenerate_energies)); Gs=Gs, Ecut_response=1)
    μ = (maximum(ε -> ε[4], data.eigenvalues)+minimum(ε -> ε[5], data.eigenvalues))/2
    f = [2 ./ (1 .+ exp.((ε.-μ)./0.03)) for ε in data.eigenvalues]
    derivative = [-fk.*(1 .- fk/2)/0.03 for fk in f]
    carrier_data = merge(data, (; occupation=f))
    @test_throws ErrorException compute_rpa_screening_voxel(carrier_data; Gs=Gs, Ecut_response=1)
    metal = compute_rpa_screening_voxel(carrier_data; Gs=Gs, Ecut_response=1,
                                          occupation_derivative=derivative)
    @test metal.compressibility > 0
    @test all(isfinite, metal.W)
    @test minimum(eigvals(Hermitian(metal.W[:, :, 1]))) > 0
end
