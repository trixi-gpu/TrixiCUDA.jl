#include("header.jl") # Remove it after first run to avoid recompilation

# Set random seed
Random.seed!(12345)

# The header part of test
advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level=4, n_cells_max=30_000)
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)

# Unpack to get key elements
@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

# Create pesudo `u`, `du` and `t` for test
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

# Rewrite `rhs!()` from `trixi/src/solvers/dgsem_tree/dg_2d.jl`
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

# CUDA kernel for calculating fluxes along normal direction 1 and 2
function flux_kernel!(flux_arr1, flux_arr2, u, equations::AbstractEquations{2}, flux::Function)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u, 1) && j <= size(u, 2)^2 && k <= size(u, 4))
        j1 = div(j, size(u, 2)) + 1
        j2 = rem(j, size(u, 2)) + 1
        @inbounds flux_arr1[i, j1, j2, k] = flux(u[i, j1, j2, k], 1, equations)
        @inbounds flux_arr2[i, j1, j2, k] = flux(u[i, j1, j2, k], 2, equations)
    end

    return nothing
end

# Calculate volume integral
function cuda_volume_integral!(du, u,
    mesh::TreeMesh{2},
    nonconservative_terms, equations,
    volume_integral::VolumeIntegralWeakForm,
    dg::DGSEM, cache)

    derivative_dhat = CuArray{Float32}(dg.basis.derivative_dhat)
    flux_arr = similar(u)

end


# Inside `rhs!()` raw implementation
#################################################################################
du, u = copy_to_gpu!(du, u)

flux_arr1 = similar(u)
flux_arr2 = similar(u)
size_arr = CuArray{Float32}(undef, size(u, 1), size(u, 2)^2, size(u, 4))

#= @benchmark begin
    flux1_kernel = @cuda launch = false flux1_kernel!(flux_arr1, u, equations, flux)
    flux1_kernel(flux_arr1, u, equations, flux; configurator_3d(flux1_kernel, size_arr)...)
    flux2_kernel = @cuda launch = false flux2_kernel!(flux_arr2, u, equations, flux)
    flux2_kernel(flux_arr2, u, equations, flux; configurator_3d(flux2_kernel, size_arr)...)
end =#


@benchmark begin
    flux_kernel = @cuda launch = false flux_kernel!(flux_arr1, flux_arr2, u, equations, flux)
    flux_kernel(flux_arr1, flux_arr2, u, equations, flux; configurator_3d(flux_kernel, size_arr)...)
end


#################################################################################

#= nelements(dg::DG, cache) = nelements(cache.elements) =#

#= nelements(elements::ElementContainer2D) = length(elements.cell_ids) =#

#= ntuple(_ -> StaticInt(nnodes(solver)), ndims(mesh))..., nelements(solver, cache) =#

#= unsafe_wrap(Array{eltype(u_ode),ndims(mesh) + 2}, pointer(u_ode),
    (nvariables(equations), ntuple(_ -> nnodes(solver), ndims(mesh))..., nelements(solver, cache))) =#