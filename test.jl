# using CUDA

# # Kernel using CG.this_grid() for grid-level synchronization
# function simple_kernel(data)
#     idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

#     # Simple operation: increment each element
#     if idx <= length(data)
#         data[idx] += 1.0f0
#     end

#     grid = CG.this_grid()
#     CG.sync(grid)

#     return nothing
# end

# # Host code
# function main()
#     n = 1024
#     data = CUDA.fill(Float32(1.0), n)  # Initialize array with 1.0s

#     # Kernel launch configuration
#     threads_per_block = 10
#     num_blocks = ceil(Int, n / threads_per_block)

#     println("num_blocks: ", num_blocks)

#     # Launch kernel with cooperative launch enabled
#     kernel = @cuda launch=false simple_kernel(data)

#     kernel(data; blocks=num_blocks, threads=threads_per_block, cooperative=true)

#     # Fetch results back to host
#     result = Array(data)
# end

# main()

using CUDA

function query_max_threads_per_block()
    device = CUDA.device()  # Get the current CUDA device
    max_threads = CUDA.capability(device).max_threads_per_block  # Query max threads per block
    println("Maximum threads per block: $max_threads")
end

query_max_threads_per_block()
