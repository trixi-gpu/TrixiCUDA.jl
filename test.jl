using CUDA, Test, BenchmarkTools

function foo!(a, b)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (i <= 4)
        @inbounds begin
            for ii in 1:4
                a[ii] += b[ii]
            end
        end
    end

    return nothing
end

a = CUDA.ones(4)
b = CUDA.ones(4)
@cuda threads = 4 foo!(a, b)