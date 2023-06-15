mesh = TreeMesh(coordinates_min, coordinates_max,
	initial_refinement_level = 2,
	n_cells_max = 30_000)

transpose([0.0; -0.5; -0.75; -0.25; 0.5; 0.25; 0.75;;])

# Rewrite `rhs!()` from `trixi/src/solvers/dgsem_tree/dg_1d.jl`
#################################################################################
function rhs!(du, u, t,
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
end
############################################################

N = 2^8
x = CUDA.ones(N, N, N)
y = CUDA.zeros(N, N, N)
@cuda threads = (68, 68, 68) blocks = (1, 1, 1) gpu_add10!(y, x)
@test all(Array(y) .== 1.0f0)