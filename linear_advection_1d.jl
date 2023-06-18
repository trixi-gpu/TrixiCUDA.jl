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

    #= @unpack derivative_dhat = dg.basis
    derivative_dhat = CuArray{Float32}(derivative_dhat) =#

    return (du, u)
end

# Copy `du` and `u` to CPU (back to Float64)
function copy_to_cpu!(du, u)
    du = Array{Float64}(du)
    u = Array{Float64}(u)

    return (du, u)
end

# Calculate flux array based on `u`
function cuda_flux!(flux_arr, u, equations::AbstractEquations)
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
    mesh::TreeMesh{1},                                  # StructuredMesh{1}? How to set arguments?
    nonconservative_terms, equations,
    volume_integral::VolumeIntegralWeakForm,
    dg::DGSEM, cache)

    @unpack derivative_dhat = solver.basis
    derivative_dhat = CuArray{Float32}(derivative_dhat)

    flux_arr = similar(u)
    @cuda threads = (1, 2, 4) blocks = (1, 2, 4) cuda_flux!(flux_arr, u, equations) # flux.(u, 1, equations)

    du_temp = reshape(permutedims(flux_arr, [1, 3, 2]), size(u, 1) * size(u, 3), :) * transpose(derivative_dhat)
    du = permutedims(reshape(du_temp, size(u, 1), size(u, 3), :), [1, 3, 2])

    return (du, u)
end

# Prolong two boundary interfaces
function cuda_prolong2interfaces!(cache, u,
    mesh::TreeMesh{1}, equations, surface_integral, dg::DG)

    @unpack interfaces = cache

    u_temp = Array(reshape(permutedims(u, [1, 3, 2]), size(u, 1) * size(u, 3), :))
    u1 = u_temp[:, end]
    u2 = vcat(u_temp[:, 1][size(u, 1)+1:end], u_temp[:, 1][1:size(u, 1)])

    interfaces_u = permutedims(reshape(hcat(u1, u2), size(u, 1), size(u, 3), :), [1, 3, 2])
    interfaces.u = permutedims(interfaces_u, [2, 1, 3])

    return nothing
end

#

# Inside `rhs!()` raw implementation
#################################################################################
du, u = copy_to_gpu!(du, u)

du, u = cuda_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    solver.volume_integral, solver, cache)

cuda_prolong2interfaces!(
    cache, u, mesh, equations, solver.surface_integral, solver)




# For tests
#= reset_du!(du, solver, cache)

calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    solver.volume_integral, solver, cache)

prolong2interfaces!(
    cache, u, mesh, equations, solver.surface_integral, solver) =#

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