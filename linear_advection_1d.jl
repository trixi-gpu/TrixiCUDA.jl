#include("header.jl") # Remove it after first run to avoid recompilation

# Set random seed
Random.seed!(1234)

# The header part of test
advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level=4, n_cells_max=30_000)
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

initial_condition_sine_wave(x, t, equations) = SVector(1.0 + 0.5 * sin(pi * sum(x - equations.advection_velocity * t)))
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sine_wave, solver)

# Unpack to get key elements
@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

# Create pesudo `u` and `du` for test
l = nvariables(equations) * nnodes(solver)^ndims(mesh) * nelements(solver, cache)
u_ode = rand(Float64, l)
du_ode = rand(Float64, l)
u = wrap_array(u_ode, mesh, equations, solver, cache)
du = wrap_array(du_ode, mesh, equations, solver, cache)

# Rewrite `rhs!()` from `trixi/src/solvers/dgsem_tree/dg_1d.jl`
#################################################################################

# Copy `du` and `u` to GPU (run as Float32)
function copy_to_gpu!(du, u)
    du = CuArray{Float32}(du)
    u = CuArray{Float32}(u)

    return (du, u)
end

# Copy `du` and `u` to CPU (back to Float64)
function copy_to_cpu!(du, u)
    du = Array{Float64}(du)
    u = Array{Float64}(u)

    return (du, u)
end

# Calculate flux based on `u`
function cuda_flux!(flux_arr, u, equations::AbstractEquations, flux::Function)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u, 1) && j <= size(u, 2) && k <= size(u, 3))
        @inbounds flux_arr[i, j, k] = flux(u[i, j, k], 1, equations)
    end

    return nothing
end

# Calculate volume integral
function cuda_volume_integral!(du, u,
    mesh::TreeMesh{1},                                  # StructuredMesh{1}? 
    nonconservative_terms, equations,
    volume_integral::VolumeIntegralWeakForm,
    dg::DGSEM, cache)

    derivative_dhat = CuArray{Float32}(dg.basis.derivative_dhat)

    flux_arr = similar(u)
    @cuda threads = (1, 2, 4) blocks = (1, 2, 4) cuda_flux!(flux_arr, u, equations, flux) # Configurator

    du_temp = reshape(permutedims(flux_arr, [1, 3, 2]), size(u, 1) * size(u, 3), :) * transpose(derivative_dhat)
    du = permutedims(reshape(du_temp, size(u, 1), size(u, 3), :), [1, 3, 2])

    return du
end

# Prolong solution to interfaces
function cuda_prolong2interfaces!(cache, u,
    mesh::TreeMesh{1}, equations, surface_integral, dg::DG)

    u_temp = reshape(permutedims(u, [1, 3, 2]), size(u, 1) * size(u, 3), :)
    u1 = u_temp[:, end]
    u2 = vcat(u_temp[:, 1][size(u, 1)+1:end], u_temp[:, 1][1:size(u, 1)])

    interfaces_u = permutedims(reshape(hcat(u1, u2), size(u, 1), size(u, 3), :), [1, 3, 2])
    cache.interfaces.u = permutedims(interfaces_u, [2, 1, 3])  # Automatically copy back to CPU

    return nothing
end

# Calculate surface flux based on `cache.interfaces.u`
function cuda_surface_flux!(surface_flux_arr, u, equations::AbstractEquations, surface_flux::FluxLaxFriedrichs) # Other fluxes?
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2) && k <= size(u, 3))
        @inbounds surface_flux_arr[1, j, k] = surface_flux(u[1, j, k], u[2, j, k], 1, equations)
    end

    return nothing
end

# Calculate interface fluxes
function cuda_interface_flux!(cache,
    mesh::TreeMesh{1},
    nonconservative_terms::False, equations, # Skip `nonconservative_terms::True`
    surface_integral, dg::DG)

    surface_flux = surface_integral.surface_flux
    u = CuArray{Float32}(cache.interfaces.u)

    surface_flux_arr = CuArray{Float32}(undef, (1, size(u, 2), size(u, 3)))
    @cuda threads = (1, 4) blocks = (1, 4) cuda_surface_flux!(surface_flux_arr, u, equations, surface_flux) # Configurator

    surface_flux_temp = reshape(permutedims(permutedims(surface_flux_arr, [2, 1, 3]), [1, 3, 2]), :, 1)
    surface_flux1 = surface_flux_temp
    surface_flux2 = vcat(surface_flux_temp[end-size(u, 2)+1:end], surface_flux_temp[1:end-size(u, 2)])

    cache.elements.surface_flux_values = permutedims(reshape(hcat(surface_flux2, surface_flux1), size(u, 2), size(u, 3), :), [1, 3, 2])

    return nothing
end

# Prolong solution to boundaries
# Calculate boundary fluxes

# # Calculate surface integrals
function cuda_surface_integral!(du, u, mesh::TreeMesh{1},           # StructuredMesh{1}? 
    equations, surface_integral, dg::DGSEM, cache)

    factor1 = dg.basis.boundary_interpolation[1, 1]
    factor2 = dg.basis.boundary_interpolation[size(u, 2), 2]

    surface_flux_values = CuArray{Float32}(cache.elements.surface_flux_values)
    du_temp = reshape(permutedims(du, [1, 3, 2]), size(u, 1) * size(u, 3), :)

    surface_flux_temp = reshape(permutedims(surface_flux_values, [1, 3, 2]), size(u, 1) * size(u, 3), :)
    surface_integral1 = factor1 .* surface_flux_temp[:, 1]
    surface_integral2 = factor2 .* surface_flux_temp[:, 2]

    du_temp[:, 1] -= surface_integral1
    du_temp[:, end] += surface_integral2

    du = permutedims(reshape(du_temp, size(u, 1), size(u, 3), :), [1, 3, 2])

    return du
end

# Apply Jacobian from mapping to reference element
function cuda_jacobian!(du, mesh::TreeMesh{1},                 # StructuredMesh{1}?
    equations, dg::DG, cache)

    inverse_jacobian = -CuArray{Float32}(cache.elements.inverse_jacobian)
    factor_arr = similar(du)
    factor_arr .= reshape(inverse_jacobian, 1, 1, :)

    du .*= factor_arr

    return du
end

# Calculate source terms              Overhead?
function cuda_sources!(du, u, t, source_terms::Nothing, # Skip `source_terms` has something
    equations::AbstractEquations{1}, dg::DG, cache)
    return nothing
end

# Inside `rhs!()` raw implementation
#################################################################################
du, u = copy_to_gpu!(du, u)

du = cuda_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    solver.volume_integral, solver, cache)

cuda_prolong2interfaces!(
    cache, u, mesh, equations, solver.surface_integral, solver)

cuda_interface_flux!(
    cache, mesh,
    have_nonconservative_terms(equations), equations,
    solver.surface_integral, solver)

#= cuda_prolong2boundaries!(
    cache, u, mesh, equations, solver.surface_integral, solver) =#

#= cuda_boundary_flux!(
    cache, t, boundary_conditions, mesh,
    equations, solver.surface_integral, solver) =#

du = cuda_surface_integral!(
    du, u, mesh, equations, solver.surface_integral, solver, cache)

du = cuda_jacobian!(
    du, mesh, equations, solver, cache)

#= cuda_sources!(du, u, t,
    source_terms, equations, solver, cache) =#

du, u = copy_to_cpu!(du, u)







# For tests
#= reset_du!(du, solver, cache)

calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    solver.volume_integral, solver, cache)

prolong2interfaces!(
    cache, u, mesh, equations, solver.surface_integral, solver)

calc_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    solver.surface_integral, solver, cache)

calc_surface_integral!(
    du, u, mesh, equations, solver.surface_integral, solver, cache)

apply_jacobian!(
    du, mesh, equations, solver, cache) =#

#################################################################################

#= len = nelements(solver, cache)
kernel = @cuda name = "copy to" launch = false copy_to_gpu!(du, len)
kernel(du, u, solver, cache; configurator(kernel, nelements(dg, cache))...) =#

#= # Configure block and grid for kernel
function configurator(kernel::CUDA.HostKernel, length::Integer)  # for 1d
	config = launch_configuration(kernel.fun)
	threads = min(length, config.threads)
	blocks = cld(length, threads) # min(attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X), cld(length, threads))
	return (threads = threads, blocks = blocks)
end =#
