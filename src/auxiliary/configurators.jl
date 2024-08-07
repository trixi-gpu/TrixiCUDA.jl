# Kernel configurators

# Kernel configurator for 1D CUDA array
function configurator_1d(kernel::HostKernel, array::CuArray{<:Any, 1})
    config = launch_configuration(kernel.fun)

    threads = min(length(array), config.threads)
    blocks = cld(length(array), threads)

    return (threads = threads, blocks = blocks)
end

# Kernel configurator for 2D CUDA array 
function configurator_2d(kernel::HostKernel, array::CuArray{<:Any, 2})
    config = launch_configuration(kernel.fun)

    threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 2))),
                         2))
    blocks = map(cld, size(array), threads)

    return (threads = threads, blocks = blocks)
end

# Kernel configurator for 3D CUDA array
function configurator_3d(kernel::HostKernel, array::CuArray{<:Any, 3})
    config = launch_configuration(kernel.fun)

    threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 3))),
                         3))
    blocks = map(cld, size(array), threads)

    return (threads = threads, blocks = blocks)
end
