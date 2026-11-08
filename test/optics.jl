@testitem "Optics: kinetic velocity and Drude limit" tags=[:optics] begin
    using DFTK
    using LinearAlgebra

    model = Model(8.0Matrix(I,3,3); n_electrons=2, terms=[Kinetic()],
                  symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=1,
                           kgrid=ExplicitKpoints([[0.13,0.07,-0.04]]))
    kpt = only(basis.kpoints)
    ε = [sum(abs2,p)/2 for p in Gplusk_vectors_cart(basis,kpt)]
    order = sortperm(ε)
    ψ = Matrix{ComplexF64}(I,length(ε),length(ε))[:,order]
    data = (; basis, ψ=[ψ], eigenvalues=[ε[order]],
             occupation=[zeros(length(ε))], ρ=guess_density(basis))
    velocity = compute_velocity(data)
    masses = compute_effective_mass(data;band=1)
    @test masses.inverse_mass ≈ I atol=1e-12
    @test masses.principal_masses ≈ ones(3)
    for α = 1:3
        @test velocity[1][:,:,α] ≈ Diagonal(getindex.(
            Gplusk_vectors_cart(basis,kpt),α)[order]) atol=1e-13
    end
    derivative = [-ones(length(ε))]
    ωp2 = compute_drude_weight(basis,velocity,derivative)
    ω = [0.2,0.3]
    spectrum = compute_dielectric(data,ω; velocity, η=0.01,
                                   occupation_derivative=derivative)
    for iω in eachindex(ω)
        @test spectrum.ε[:,:,iω] ≈ I - ωp2/(ω[iω]*(ω[iω]+0.01im))
    end
    @test_throws ArgumentError compute_dielectric(data,[0.0]; velocity,η=0.01,
                                                  occupation_derivative=derivative)
end

@testitem "Optics: degenerate Drude subspace covariance" tags=[:optics] begin
    using DFTK
    using LinearAlgebra
    model = Model(5.0Matrix(I,3,3);n_electrons=2,terms=[Kinetic()],symmetries=false)
    basis = PlaneWaveBasis(model;Ecut=1,kgrid=[1,1,1])
    v = zeros(ComplexF64,2,2,3)
    v[:,:,1] = [0 1;1 0]
    v[:,:,2] = [1 0;0 -1]
    derivative = [[-0.5,-0.5]]
    ε = [[0.0,0.0]]
    f = [[1.0,1.0]]
    D = compute_drude_weight(basis,[v],derivative;eigenvalues=ε)
    @test D ≈ 4π/model.unit_cell_volume*diagm([1.0,1.0,0.0])
    U = [1 im;im 1]/sqrt(2)
    rotated = cat([U'*v[:,:,α]*U for α = 1:3]...;dims=3)
    @test compute_drude_weight(basis,[rotated],derivative;eigenvalues=ε) ≈ D
    ω = [0.1,0.2]
    original = compute_dielectric(basis,ε,f,[v],ω;η=0.01,occupation_derivative=derivative)
    changed = compute_dielectric(basis,ε,f,[rotated],ω;η=0.01,occupation_derivative=derivative)
    @test changed.ε ≈ original.ε
    δ = 1e-10
    εnear = [[0.0,δ]]
    fnear = [[1.0,2Smearing.occupation(Smearing.FermiDirac(),δ)]]
    near = compute_dielectric(basis,εnear,fnear,[v],ω;
                               η=0.01,occupation_derivative=derivative)
    @test near.ε ≈ original.ε
    @test_throws ArgumentError compute_dielectric(basis,ε,[[2.0,0.0]],[v],ω;
        η=0.01,occupation_derivative=derivative)
end

@testitem "Optics: Γ-point projector origin limits" tags=[:optics] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    Si = ElementPsp(:Si,TestCases.pd_lda_family)
    model = model_atomic(TestCases.silicon.lattice,[Si,Si],TestCases.silicon.positions;
                          symmetries=false)
    basis = PlaneWaveBasis(model;Ecut=5,kgrid=[1,1,1])
    ρ = guess_density(basis)
    F = eigen(Hermitian(Matrix(Hamiltonian(basis;ρ)[1])))
    data = (;basis,ρ,ψ=[F.vectors[:,1:10]],eigenvalues=[F.values[1:10]],
             occupation=[vcat(fill(2.0,4),zeros(6))])
    velocity = compute_velocity(data)[1]
    @test all(isfinite,velocity)
    kpt = only(basis.kpoints)
    for α = 1:3
        h = 1e-5
        direction = Matrix{Float64}(I,3,3)[:,α]
        function projected(δk)
            shifted = PlaneWaveBasis(basis,ExplicitKpoints([
                basis.model.inv_recip_lattice*δk]))
            @test only(shifted.kpoints).G_vectors == kpt.G_vectors
            data.ψ[1]'*Matrix(Hamiltonian(shifted;ρ)[1])*data.ψ[1]
        end
        @test velocity[:,:,α] ≈ (projected(h*direction)-projected(-h*direction))/(2h) #=
            =# atol=2e-8 rtol=1e-6
    end
    @test_throws ErrorException compute_effective_mass(data;band=1)
end

@testitem "Optics: Hubbard projector k derivative" tags=[:optics] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    Si = ElementPsp(:Si,TestCases.pd_lda_family)
    hubbard = Hubbard(OrbitalManifold(Si,(;l=0,i=1)) => 0.2)
    model = model_atomic(TestCases.silicon.lattice,[Si,Si],TestCases.silicon.positions;
                         extra_terms=[hubbard],symmetries=false)
    basis = PlaneWaveBasis(model;Ecut=5,
                           kgrid=ExplicitKpoints([[0.13,0.07,-0.04]]))
    ρ = guess_density(basis)
    hubbard_n = DFTK.guess_hubbard_n(basis)
    for n in only(hubbard_n)
        n .*= 0.4
    end
    ham = Hamiltonian(basis;ρ,hubbard_n)
    F = eigen(Hermitian(Matrix(ham[1])))
    data = (;basis,ρ,hubbard_n,ψ=[F.vectors[:,1:8]],eigenvalues=[F.values[1:8]],
             occupation=[vcat(fill(2.0,4),zeros(4))])
    velocity = compute_velocity(data)[1]
    @test all(isfinite,velocity)
    k = only(basis.kpoints).coordinate
    for α = 1:3
        δ = 1e-5
        δk = basis.model.inv_recip_lattice[:,α]*δ
        function bands(k)
            shifted = PlaneWaveBasis(basis,ExplicitKpoints([k]))
            eigvals(Hermitian(Matrix(Hamiltonian(shifted;ρ,hubbard_n)[1])))[1:8]
        end
        @test real.(diag(velocity[:,:,α])) ≈ (bands(k+δk)-bands(k-δk))/(2δ) #=
            =# atol=2e-6 rtol=2e-5
    end
end

@testitem "Optics: nonlocal velocity versus band slopes" tags=[:optics] #=
    =# setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    using .TestCases: silicon

    model = model_atomic(silicon.lattice,silicon.atoms,silicon.positions;
                         symmetries=false)
    k = [0.13,0.07,-0.04]
    basis = PlaneWaveBasis(model; Ecut=5,kgrid=ExplicitKpoints([k]))
    ρ = guess_density(basis)
    ham = Hamiltonian(basis; ρ)
    F = eigen(Hermitian(Matrix(ham[1])))
    data = (; basis,ρ,ψ=[F.vectors[:,1:10]],eigenvalues=[F.values[1:10]],
             occupation=[vcat(fill(2.0,4),zeros(6))])
    velocity = compute_velocity(data)[1]
    @test all(isfinite,velocity)
    for α = 1:3
        δ = 1e-5
        δk = basis.model.inv_recip_lattice[:,α] * δ
        function bands(k)
            shifted = PlaneWaveBasis(basis,ExplicitKpoints([k]))
            eigvals(Hermitian(Matrix(Hamiltonian(shifted;ρ)[1])))[1:10]
        end
        slope = (bands(k+δk)-bands(k-δk))/(2δ)
        @test real.(diag(velocity[:,:,α])) ≈ slope atol=2e-6 rtol=2e-5
        @test velocity[:,:,α] ≈ velocity[:,:,α]' atol=1e-12
    end
    complete = merge(data,(;ψ=[F.vectors],eigenvalues=[F.values],
                            occupation=[vcat(fill(2.0,4),zeros(length(F.values)-4))]))
    masses = compute_effective_mass(complete;band=1)
    for α = 1:3
        h = 2e-4
        δk = basis.model.inv_recip_lattice[:,α]*h
        function ε1(k)
            shifted = PlaneWaveBasis(basis,ExplicitKpoints([k]))
            minimum(eigvals(Hermitian(Matrix(Hamiltonian(shifted;ρ)[1]))))
        end
        curvature = (ε1(k+δk)-2ε1(k)+ε1(k-δk))/h^2
        @test masses.inverse_mass[α,α] ≈ curvature atol=5e-6 rtol=2e-4
    end
    # Independently differentiate all matrix elements at a fixed G set.
    ψ = data.ψ[1]
    for α = 1:3
        δ = 1e-5
        direction = Matrix{Float64}(I,3,3)[:,α]
        function projected(δk)
            shifted = DFTK._shift_kpoint(basis,only(basis.kpoints),δk)
            sum(zip(model.term_types,basis.terms)) do (builder,term)
                DFTK._optical_kmatrix(builder,term,basis,shifted,ψ;hubbard_n=nothing)
            end
        end
        @test velocity[:,:,α] ≈ (projected(δ*direction)-projected(-δ*direction))/(2δ) #=
            =# atol=1e-8 rtol=1e-6
    end
end

@testitem "Optics: causal two-level spectrum and sum rule" tags=[:optics] begin
    using DFTK
    using LinearAlgebra

    model = Model(5.0Matrix(I,3,3); n_electrons=2,terms=[Kinetic()],symmetries=false)
    basis = PlaneWaveBasis(model;Ecut=1,kgrid=[1,1,1])
    Δ, v, η = 0.4, 0.3, 0.002
    velocity = zeros(ComplexF64,2,2,3)
    velocity[1,2,1] = velocity[2,1,1] = v
    ε, f = [[0.0,Δ]], [[2.0,0.0]]
    ω = collect(range(0,4.0;length=40001))
    spectrum = compute_dielectric(basis,ε,f,[velocity],ω;η)
    prefactor = 4π/model.unit_cell_volume
    analytic = @. 1 + prefactor*2v^2/Δ^2 * (1/(Δ-ω-im*η)+1/(Δ+ω+im*η))
    @test spectrum.ε[1,1,:] ≈ analytic
    @test minimum(imag.(analytic)) ≥ -1e-14
    @test spectrum.ε[2,2,:] == ones(length(ω))
    # ∫₀∞ ω Im ε(ω) dω = π/2 times the high-frequency oscillator strength.
    y = ω .* imag.(analytic)
    integral = sum((y[1:end-1]+y[2:end])/2) * step(range(0,4.0;length=40001))
    @test integral ≈ prefactor*2π*v^2/Δ rtol=0.001
    shifted = compute_dielectric(basis,ε,f,[velocity],[0.0];η,scissor=0.1,n_valence=1)
    @test real(shifted.ε[1,1,1]) < real(spectrum.ε[1,1,1])
    inverted = compute_dielectric(basis,ε,reverse.(f),[velocity],[Δ];η)
    @test imag(inverted.ε[1,1,1]) < 0
    @test_throws ArgumentError compute_dielectric(basis,ε,f,[velocity],[Δ];η=-η)
    @test_throws ArgumentError compute_dielectric(basis,ε,f,[velocity],[Δ];η,scissor=0.1)
    @test_throws ArgumentError compute_dielectric(basis,[[0.0,0.0]],f,[velocity],[Δ];η)
    @test refractive_index(4.0) == (;n=2.0,κ=0.0)
    @test absorption_coefficient(0.2,4.0) == 0
    @test loss_function(1+im) == 0.5
    x = collect(range(1,3;length=51))
    α = sqrt.(max.(x.-1.3,0)) ./ x
    fit = fit_tauc(x,α;window=(1.5,2.5),exponent=2)
    @test fit.gap ≈ 1.3 atol=1e-12
    @test fit.r2 ≈ 1
end
