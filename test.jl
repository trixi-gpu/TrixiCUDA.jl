using CUDA, Test, BenchmarkTools

function gpu_add1!(y, x)
    index = threadIdx().x
    stride = blockDim().x

    for i ∈ index:stride:length(y)
        @inbounds y[i] += x[i]
    end

    return nothing
end

function gpu_add6!(y, x)
    index_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    stride_i = gridDim().x * blockDim().x
    stride_j = gridDim().y * blockDim().y

    for i ∈ index_i:stride_i:size(y, 1)
        for j ∈ index_j:stride_j:size(y, 2)
            @inbounds y[i, j] += x[i, j]
        end
    end

    return nothing
end

function gpu_add10!(y, x)
    index_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    index_k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    stride_i = gridDim().x * blockDim().x
    stride_j = gridDim().y * blockDim().y
    stride_k = gridDim().z * blockDim().z

    for i ∈ index_i:stride_i:size(y, 1)
        for j ∈ index_j:stride_j:size(y, 2)
            for k ∈ index_k:stride_k:size(y, 3)
                @inbounds y[i, j, k] += x[i, j, k]
            end
        end
    end

    return nothing
end

# min(attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X), cld(length, threads))

const MAX_GRID_DIM_X = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X) # may not be used ???

function configurator(kernel::CUDA.HostKernel, length::Integer)  # for 1d
    config = launch_configuration(kernel.fun)
    threads = min(length, config.threads)
    println(config.blocks)
    blocks = cld(length, threads)
    println((threads, blocks))
    return (threads=threads, blocks=blocks)
end

N = 1000000
x6 = CUDA.ones(10, N)
y6 = CUDA.zeros(10, N)

kernel6 = @cuda launch = false gpu_add6!(y6, x6)
kernel6(y6, x6; configurator(kernel6, N)...)

#= @benchmark @cuda threads = 100 blocks = 2 gpu_add6!(y, x) =#

x1 = CUDA.ones(N)
y1 = CUDA.zeros(N)
kernel1 = @cuda launch = false gpu_add6!(y1, x1)
kernel1(y1, x1; configurator(kernel1, N)...)

x10 = CUDA.ones(10, 10, N)
y10 = CUDA.zeros(10, 10, N)
kernel10 = @cuda launch = false gpu_add6!(y10, x10)
kernel10(y10, x10; configurator(kernel10, N)...)