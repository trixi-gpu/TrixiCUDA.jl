module TestKernelTypeInference

using CUDA
using Test

# A small diagnostic example, not a reproduction of the multicomponent bug.
# Keep the call chain identical for both compiler settings.
function make_values(x::T) where {T}
    scale = T(1.1691)
    return ntuple(i -> scale * x + T(i), Val(4))
end

function sum_values(x::T) where {T}
    values = make_values(x)
    return values[1] + values[2] + values[3] + values[4]
end

function inspect_kernel!(output, input)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(input)
        @inbounds output[i] = sum_values(input[i])
    end
    return nothing
end

@testset "GPU kernel type inference diagnostics" begin
    input_cpu = Float32[1, 2, 3, 4]
    input = CuArray(input_cpu)
    output = similar(input)
    expected = 4 * 1.1691f0 .* input_cpu .+ 10

    println("Julia: ", VERSION)
    println("CUDA.jl: ", pkgversion(CUDA))
    println("Host input type: ", typeof(input))

    for force_inline in (false, true)
        @testset "always_inline=$force_inline" begin
            println("\n========== GPU type inference: always_inline=$force_inline ==========\n")
            @device_code_warntype @cuda launch=false always_inline=force_inline inspect_kernel!(output, input)
            flush(stdout)

            # Also execute the kernel: an IR listing alone does not validate its output.
            kernel = @cuda launch=false always_inline=force_inline inspect_kernel!(output, input)
            fill!(output, Float32(NaN))
            kernel(output, input; threads = length(input), blocks = 1)
            CUDA.synchronize()
            @test isapprox(Array(output), expected; rtol = 8 * eps(Float32))
        end
    end
end

end # module
