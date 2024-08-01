using TrixiGPU, CUDA
using Test

@testset "Test auxiliary functions" begin
    @testset "CUDA congifurator 1D" begin
        function kernel_add!(a, b, c)
            i = threadIdx().x
            c[i] = a[i] + b[i]
            return
        end

        N = 256
        a = CuArray(fill(1.0f0, N))
        b = CuArray(fill(2.0f0, N))
        c = CuArray(zeros(Float32, N))

        sample_kernel = @cuda threads=N kernel_add!(a, b, c)
        @test configurator_1d(sample_kernel, a) == (threads = 256, blocks = 1)
    end
end
