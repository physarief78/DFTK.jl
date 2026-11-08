@testitem "Optics: BLAS dual products preserve ForwardDiff" tags=[:optics] begin
    using DFTK,ForwardDiff,LinearAlgebra,Random
    rng = MersenneTwister(42)
    for T in (Float32,Float64)
        A = randn(rng,Complex{T},7,5)
        B = randn(rng,Complex{T},5,4)
        function product(x,multiply)
            left = exp(im*x[1]).*A
            right = (1+x[2]^2).*B
            C = multiply(left,right)
            vcat(vec(real.(C)),vec(imag.(C)))
        end
        x = T[0.2,0.3]
        expected = ForwardDiff.jacobian(z -> product(z,*),x)
        actual = ForwardDiff.jacobian(z -> product(z,DFTK._optical_mul),x)
        @test actual ≈ expected rtol=10eps(T) atol=10eps(T)
        real_product(x,multiply) = vec(multiply((1+x[1]).*real.(A),real.(B)))
        @test ForwardDiff.jacobian(z -> real_product(z,DFTK._optical_mul),x) ≈
              ForwardDiff.jacobian(z -> real_product(z,*),x) rtol=10eps(T) atol=10eps(T)
        # Transposes, a constant real coefficient matrix, and nested duals.
        function scalar(t,multiply)
            P = exp(im*t).*A
            real(sum(multiply(multiply(P',ones(T,7,7)),P)))
        end
        @test ForwardDiff.derivative(t -> scalar(t,DFTK._optical_mul),T(0.2)) ≈
              ForwardDiff.derivative(t -> scalar(t,*),T(0.2)) atol=100eps(T)
        @test ForwardDiff.derivative(t -> ForwardDiff.derivative(
              s -> scalar(s,DFTK._optical_mul),t),T(0.2)) ≈
              ForwardDiff.derivative(t -> ForwardDiff.derivative(
              s -> scalar(s,*),t),T(0.2)) atol=100eps(T)
    end
end
