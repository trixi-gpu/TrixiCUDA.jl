include("header.jl") # Remove it after first run to avoid recompilation

# The header part of tests
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

#= function reset_du!(du, dg, cache)
	@threaded for element in eachelement(dg, cache)
		du[.., element] .= zero(eltype(du))
	end

	return du
end =#

# Rewrite reset_du!()
function copy_to_gpu!(du, u, dg, cache)
	index = threadIdx().x
	stride = blockDim().x

	for i in index:stride:nelements(dg, cache) # nelements() from `trixi/src/solvers/dgsem_tree/containers_1d.jl`
		@inbounds du[.., i] = CuArray(du[.., i])
		@inbounds u[.., i] = CuArray(u[.., i])
	end
end

# Rewrite calc_volume_integral!()


function rhs!(du, u, t,
	mesh::TreeMesh{1}, equations,
	initial_condition, boundary_conditions, source_terms::Source,
	dg::DG, cache) where {Source}

	# Rewrite reset_du!()
	threads = nelements(dg, cache)
	@cuda threads = threads copy_to_gpu!(du, u, dg, cache)

	# Rewrite calc_volume_integral!()
	# ...
end


#= function rhs!(du, u, t,
	mesh::TreeMesh{1}, equations,
	initial_condition, boundary_conditions, source_terms::Source,
	dg::DG, cache) where {Source}
	# Reset du
	@trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

	# Calculate volume integral
	@trixi_timeit timer() "volume integral" calc_volume_integral!(
		du, u, mesh,
		have_nonconservative_terms(equations), equations,
		dg.volume_integral, dg, cache)

	# Prolong solution to interfaces
	@trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
		cache, u, mesh, equations, dg.surface_integral, dg)

	# Calculate interface fluxes
	@trixi_timeit timer() "interface flux" calc_interface_flux!(
		cache.elements.surface_flux_values, mesh,
		have_nonconservative_terms(equations), equations,
		dg.surface_integral, dg, cache)

	# Prolong solution to boundaries
	@trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
		cache, u, mesh, equations, dg.surface_integral, dg)

	# Calculate boundary fluxes
	@trixi_timeit timer() "boundary flux" calc_boundary_flux!(
		cache, t, boundary_conditions, mesh,
		equations, dg.surface_integral, dg)

	# Calculate surface integrals
	@trixi_timeit timer() "surface integral" calc_surface_integral!(
		du, u, mesh, equations, dg.surface_integral, dg, cache)

	# Apply Jacobian from mapping to reference element
	@trixi_timeit timer() "Jacobian" apply_jacobian!(
		du, mesh, equations, dg, cache)

	# Calculate source terms
	@trixi_timeit timer() "source terms" calc_sources!(
		du, u, t, source_terms, equations, dg, cache)

	return nothing
end =#
############################################################
