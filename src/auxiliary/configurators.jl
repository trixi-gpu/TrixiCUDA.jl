# Kernel configurators are used for determining the number of threads and 
# blocks to be used in the kernel, which optimizes the use of GPU resources.

# 1D kernel configurator
# We hard-code 32 threads per block for 1D kernels.
function kernel_configurator_1d(kernel::HostKernel, x::Int)
    # config = launch_configuration(kernel.fun) # not used in this case

    threads = 32 # warp size is 32, if block size is less than 32, it will be padded to 32
    blocks = cld(x, threads[1])

    return (threads = threads, blocks = blocks)
end

# 1D kernel configurator for cooperative launch
# Note that cooperative kernels can only launch as many blocks as there are SMs on the device, 
# so we need to query the SM count first. Also, kernels launched with cooperative launch have 
# to use stride loops to handle the constrained launch size.
function kernel_configurator_coop_1d(kernel::HostKernel, x::Int)
    # config = launch_configuration(kernel.fun) # not used in this case

    threads = 32 # warp size is 32, if block size is less than 32, it will be padded to 32
    blocks = min(cld(x, threads), MULTIPROCESSOR_COUNT)

    return (threads = threads, blocks = blocks)
end

# 2D kernel configurator
# We hard-code 32 threads for x dimension per block, and y dimension is determined 
# by the number of threads returned by the launch configuration.
function kernel_configurator_2d(kernel::HostKernel, x::Int, y::Int)
    config = launch_configuration(kernel.fun) # get the number of threads

    # y dimension
    dims_y1 = cld(x * y, 32)
    dims_y2 = max(fld(config.threads, 32), 1)

    dims_y = min(dims_y1, dims_y2)

    # x dimension is hard-coded to warp size 32
    threads = (32, dims_y)
    blocks = (cld(x, threads[1]), cld(y, threads[2]))

    return (threads = threads, blocks = blocks)
end

# 2D kernel configurator for cooperative launch
# Note that cooperative kernels can only launch as many blocks as there are SMs on the device, 
# so we need to query the SM count first. Also, kernels launched with cooperative launch have 
# to use stride loops to handle the constrained launch size.
function kernel_configurator_coop_2d(kernel::HostKernel, x::Int, y::Int)
    config = launch_configuration(kernel.fun) # get the number of threads

    # y dimension
    dims_y1 = cld(x * y, 32)
    dims_y2 = max(fld(config.threads, 32), 1)

    dims_y = min(dims_y1, dims_y2)

    # x dimension is hard-coded to warp size 32
    threads = (32, dims_y)
    blocks_x = cld(x, threads[1])
    blocks_y = min(cld(y, threads[2]), fld(MULTIPROCESSOR_COUNT, blocks_x))

    blocks = (blocks_x, blocks_y)

    return (threads = threads, blocks = blocks)
end

# 3D kernel configurator
# We hard-code 32 threads for x dimension per block, y and z dimensions are determined 
# by the number of threads returned by the launch configuration.
function kernel_configurator_3d(kernel::HostKernel, x::Int, y::Int, z::Int)
    config = launch_configuration(kernel.fun) # get the number of threads

    # y dimension
    dims_y1 = cld(x * y, 32)
    dims_y2 = max(fld(config.threads, 32), 1)

    dims_y = min(dims_y1, dims_y2)

    # z dimension
    dims_z1 = cld(x * y * z, 32 * dims_y)
    dims_z2 = max(fld(config.threads, 32 * dims_y), 1)

    dims_z = min(dims_z1, dims_z2)

    # x dimension is hard-coded to warp size 32
    threads = (32, dims_y, dims_z)
    blocks = (cld(x, threads[1]), cld(y, threads[2]), cld(z, threads[3]))

    return (threads = threads, blocks = blocks)
end

# 3D kernel configurator for cooperative launch
# Note that cooperative kernels can only launch as many blocks as there are SMs on the device, 
# so we need to query the SM count first. Also, kernels launched with cooperative launch have 
# to use stride loops to handle the constrained launch size.
function kernel_configurator_coop_3d(kernel::HostKernel, x::Int, y::Int, z::Int)
    config = launch_configuration(kernel.fun) # get the number of threads

    # y dimension
    dims_y1 = cld(x * y, 32)
    dims_y2 = max(fld(config.threads, 32), 1)

    dims_y = min(dims_y1, dims_y2)

    # z dimension
    dims_z1 = cld(x * y * z, 32 * dims_y)
    dims_z2 = max(fld(config.threads, 32 * dims_y), 1)

    dims_z = min(dims_z1, dims_z2)

    # x dimension is hard-coded to warp size 32
    threads = (32, dims_y, dims_z)
    blocks_x = cld(x, threads[1])
    blocks_y = min(cld(y, threads[2]), fld(MULTIPROCESSOR_COUNT, blocks_x))
    blocks_z = min(cld(z, threads[3]), fld(MULTIPROCESSOR_COUNT, blocks_x * blocks_y))

    blocks = (blocks_x, blocks_y, blocks_z)

    return (threads = threads, blocks = blocks)
end

# Deprecated old kernel configurators below

# function configurator_1d(kernel::HostKernel, array::CuArray{<:Any, 1})
#     config = launch_configuration(kernel.fun)

#     threads = min(length(array), config.threads)
#     blocks = cld(length(array), threads)

#     return (threads = threads, blocks = blocks)
# end

# function configurator_2d(kernel::HostKernel, array::CuArray{<:Any, 2})
#     config = launch_configuration(kernel.fun)

#     threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 2))), 2))
#     blocks = map(cld, size(array), threads)

#     return (threads = threads, blocks = blocks)
# end

# function configurator_3d(kernel::HostKernel, array::CuArray{<:Any, 3})
#     config = launch_configuration(kernel.fun)

#     threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 3))), 3))
#     blocks = map(cld, size(array), threads)

#     return (threads = threads, blocks = blocks)
# end
