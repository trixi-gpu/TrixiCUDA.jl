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

#= function configurator_2d(kernel::CUDA.HostKernel, array::CuArray{Float32,2}) # cuArray{2}
    config = launch_configuration(kernel.fun)

    threads = Int(ceil(sqrt(min(maximum(size(array)), config.threads))))
    blocks = cld(maximum(size(array)), threads)

    return (threads=(threads, threads), blocks=(blocks, blocks))
end =#

function configurator_2d(kernel::CUDA.HostKernel, array::CuArray{Float32,2}) # cuArray{2}
    config = launch_configuration(kernel.fun)

    threads = Tuple(fill(Int(ceil((min(maximum(size(array)), config.threads))^(1 / 2))), 2))
    blocks = map(cld, size(array), threads)

    println((threads, blocks))
    return (threads=threads, blocks=blocks)
end


#= x6 = CUDA.ones(10, 10000)
y6 = CUDA.zeros(10, 10000)
kernel6 = @cuda launch = false gpu_add6!(y6, x6)
kernel6(y6, x6; configurator_2d(kernel6, y6)...) =#


function configurator_3d(kernel::CUDA.HostKernel, array::CuArray{Float32,3})
    config = launch_configuration(kernel.fun)

    threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 3))), 3))
    blocks = map(cld, size(array), threads)

    println((threads, blocks, config.threads))

    return (threads=threads, blocks=blocks)
end

x10 = CUDA.ones(10, 10, 1000)
y10 = CUDA.zeros(10, 10, 1000)
kernel10 = @cuda launch = false gpu_add10!(y10, x10)
kernel10(y10, x10; configurator_3d(kernel10, y10)...)

#= @cuda threads = (4, 4, 4) blocks = (10, 10, 200) gpu_add10!(y10, x10) =#

A = CUDA.rand(2, 2)
B = CUDA.rand(2, 2)

function foo!(A, B)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    a = A[i, :]
    for ii in 1:2
        B[i, j] = a[ii]
    end
    return nothing
end

@cuda threads = (2, 2) blocks = (1, 1) foo!(A, B)