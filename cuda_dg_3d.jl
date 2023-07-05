# Remove it after first run to avoid recompilation
#include("header.jl")

# Use the target test header file
include("test/linear_scalar_advection_2d.jl")

# Kernel configurators 
#################################################################################

# CUDA kernel configurator for 1D array computing
function configurator_1d(kernel::CUDA.HostKernel, array::CuArray{Float32,1})
    config = launch_configuration(kernel.fun)

    threads = min(length(array), config.threads)
    blocks = cld(length(array), threads)

    return (threads=threads, blocks=blocks)
end

# CUDA kernel configurator for 2D array computing
function configurator_2d(kernel::CUDA.HostKernel, array::CuArray{Float32,2})
    config = launch_configuration(kernel.fun)

    threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 2))), 2))
    blocks = map(cld, size(array), threads)

    return (threads=threads, blocks=blocks)
end

# CUDA kernel configurator for 3D array computing
function configurator_3d(kernel::CUDA.HostKernel, array::CuArray{Float32,3})
    config = launch_configuration(kernel.fun)

    threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 3))), 3))
    blocks = map(cld, size(array), threads)

    return (threads=threads, blocks=blocks)
end

# Helper functions
#################################################################################

# Rewrite `get_node_vars()` as a helper function
@inline function get_nodes_vars(u, equations, indices...)

    SVector(ntuple(@inline(v -> u[v, indices...]), Val(nvariables(equations))))
end

# Rewrite `get_surface_node_vars()` as a helper function
@inline function get_surface_node_vars(u, equations, indices...)

    u_ll = SVector(ntuple(@inline(v -> u[1, v, indices...]), Val(nvariables(equations))))
    u_rr = SVector(ntuple(@inline(v -> u[2, v, indices...]), Val(nvariables(equations))))

    return u_ll, u_rr
end

# Rewrite `get_node_coords()` as a helper function
@inline function get_node_coords(x, equations, indices...)

    SVector(ntuple(@inline(idx -> x[idx, indices...]), Val(ndims(equations))))
end

# CUDA kernels 
#################################################################################

# Copy data to GPU (run as Float32)
function copy_to_gpu!(du, u)
    du = CUDA.zeros(size(du))
    u = CuArray{Float32}(u)

    return (du, u)
end

# Copy data to CPU (back to Float64)
function copy_to_cpu!(du, u)
    du = Array{Float64}(du)
    u = Array{Float64}(u)

    return (du, u)
end
