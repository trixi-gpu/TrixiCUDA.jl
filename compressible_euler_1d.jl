#include("header.jl") # Remove it after first run to avoid recompilation

# Set random seed
Random.seed!(12345)

# The header part of test
equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test
solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs)

coordinates_min = 0.0
coordinates_max = 2.0
mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=4,
    n_cells_max=10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
    source_terms=source_terms_convergence_test)

# Unpack to get key elements
@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

# Create pesudo `u`, `du` and `t` for tests
t = 0.0
l = nvariables(equations) * nnodes(solver)^ndims(mesh) * nelements(solver, cache)
u_ode = rand(Float64, l)
du_ode = rand(Float64, l)
u = wrap_array(u_ode, mesh, equations, solver, cache)
du = wrap_array(du_ode, mesh, equations, solver, cache)

# CUDA kernel configurators for 1D, 2D, and 3D arrays
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

# Rewrite `rhs!()` from `trixi/src/solvers/dgsem_tree/dg_1d.jl`
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

# CUDA kernel for calculating fluxes along normal direction 1 
function flux_kernel!(flux_arr, u, equations::AbstractEquations{1}, flux::Function)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u, 1) && j <= size(u, 2) && k <= size(u, 3))

        # Cause unsupported call errors
        u_node = zeros(size(u, 1))
        for ii in axes(u, 1)
            u_node[ii] = u[ii, j, k]
        end
        u_node = SVector{size(u, 1),Float64}(u_node)

        # Works for constant variable
        #= u_node = SVector(1.0, 2.0, 3.0) =#

        @inbounds flux_arr[i, j, k] = flux(u_node, 1, equations)[i]
    end

    return nothing
end

# CUDA kernel for calculating weak form
function weak_form_kernel!(du, derivative_dhat, flux_arr)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        for ii in 1:size(du, 2)
            du[i, j, k] += derivative_dhat[j, ii] * flux_arr[i, ii, k]
        end
    end

    return nothing
end

# Calculate volume integral
function cuda_volume_integral!(du, u,
    mesh::TreeMesh{1},
    nonconservative_terms, equations,                         # Need ...?
    volume_integral::VolumeIntegralWeakForm,
    dg::DGSEM, cache)

    derivative_dhat = CuArray{Float32}(dg.basis.derivative_dhat)
    flux_arr = similar(u)

    @cuda threads = (4, 4, 4) blocks = (4, 4, 4) flux_kernel!(flux_arr, u, equations, flux)
    #= @cuda threads = (4, 4, 4) blocks = (4, 4, 4) weak_form_kernel!(du, derivative_dhat, flux_arr) =#

    return nothing
end


# Inside `rhs!()` raw implementation
#################################################################################
du, u = copy_to_gpu!(du, u)

derivative_dhat = CuArray{Float32}(solver.basis.derivative_dhat)
flux_arr = similar(u)

@cuda threads = (3, 5, 4) blocks = (1, 1, 4) flux_kernel!(flux_arr, u, equations, flux)
#= @cuda threads = (4, 4, 4) blocks = (4, 4, 4) weak_form_kernel!(du, derivative_dhat, flux_arr) =#
