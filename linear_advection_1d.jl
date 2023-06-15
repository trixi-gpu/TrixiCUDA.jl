#include("header.jl") # Remove it after first run to avoid recompilation

# The header part of test
advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level = 4, n_cells_max = 30_000)
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

initial_condition_sine_wave(x, t, equations) = SVector(1.0 + 0.5 * sin(pi * sum(x - equations.advection_velocity * t)))
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sine_wave, solver)

# Unpack to get key elements
@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

# Create pesudo `u` and `du` for test
l = nvariables(equations) * nnodes(solver)^ndims(mesh) * nelements(solver, cache)
u_ode = Array{Float64}(undef, l)
du_ode = Array{Float64}(undef, l)
u = wrap_array(u_ode, mesh, equations, solver, cache)
du = wrap_array(du_ode, mesh, equations, solver, cache)

# Rewrite `rhs!()` from `trixi/src/solvers/dgsem_tree/dg_1d.jl`
#################################################################################

# Configure block and grid for kernel
function configurator(kernel::CUDA.HostKernel, length::Integer)  # for 1d
	config = launch_configuration(kernel.fun)
	threads = min(length, config.threads)
	blocks = cld(length, threads) # min(attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X), cld(length, threads))
	return (threads = threads, blocks = blocks)
end

# Copy `du` and `u` to GPU
function copy_to_gpu!(du::PtrArray, length::Integer)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

	if i <= length # nelements() from `trixi/src/solvers/dgsem_tree/containers_1d.jl`
		@inbounds du[.., i] = CuArray(du[.., i])
		#= @inbounds u[.., i] = CuArray(u[.., i]) =#
	end

	return nothing
end

u_ode
# Copy `du` and `u` to CPU

#= len = nelements(solver, cache)
kernel = @cuda name = "copy to" launch = false copy_to_gpu!(du, len)
kernel(du, u, solver, cache; configurator(kernel, nelements(dg, cache))...) =#


#= function rhs!(du, u, t,
	mesh::TreeMesh{1}, equations,
	initial_condition, boundary_conditions, source_terms::Source,
	dg::DG, cache) where {Source}

	# Copy data to GPU
	threads = nelements(dg, cache)
	@cuda threads = threads copy_to_gpu!(du, u, dg, cache)

	# Rewrite calc_volume_integral!()
	# ...
end =#

