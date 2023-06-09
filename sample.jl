# Rewrite sample based on code snippets from dg_1d.jl
# Just a sample, no guarantee that it can run without errors!


function rhs!(du, u, t,
	mesh::TreeMesh{1}, equations,
	initial_condition, boundary_conditions, source_terms::Source,
	dg::DG, cache, gpu = true) where {Source} # add one more argument for the outer function
	# Reset du
	@trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache, gpu = true) # add one more argument for the inner function, all following functions are similar

	# Calculate volume integral
	@trixi_timeit timer() "volume integral" calc_volume_integral!(
		du, u, mesh,
		have_nonconservative_terms(equations), equations,
		dg.volume_integral, dg, cache, gpu = true) # add argument

	# Prolong solution to interfaces
	@trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
		cache, u, mesh, equations, dg.surface_integral, dg, gpu = true) # add argument

	# Calculate interface fluxes
	@trixi_timeit timer() "interface flux" calc_interface_flux!(
		cache.elements.surface_flux_values, mesh,
		have_nonconservative_terms(equations), equations,
		dg.surface_integral, dg, cache, gpu = true) # add argument

	# Prolong solution to boundaries
	@trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
		cache, u, mesh, equations, dg.surface_integral, dg, gpu = true) # add argument

	# Calculate boundary fluxes
	@trixi_timeit timer() "boundary flux" calc_boundary_flux!(
		cache, t, boundary_conditions, mesh,
		equations, dg.surface_integral, dg, gpu = true) # add argument

	# Calculate surface integrals
	@trixi_timeit timer() "surface integral" calc_surface_integral!(
		du, u, mesh, equations, dg.surface_integral, dg, cache, gpu = true) # add argument

	# Apply Jacobian from mapping to reference element
	@trixi_timeit timer() "Jacobian" apply_jacobian!(
		du, mesh, equations, dg, cache, gpu = true) # add argument

	# Calculate source terms
	@trixi_timeit timer() "source terms" calc_sources!(
		du, u, t, source_terms, equations, dg, cache, gpu = true) # add argument

	return nothing
end

function calc_volume_integral!(du, u,
	mesh::Union{TreeMesh{1}, StructuredMesh{1}},
	nonconservative_terms, equations,
	volume_integral::VolumeIntegralWeakForm,
	dg::DGSEM, cache, gpu = true) # add argument

	@threaded for element in eachelement(dg, cache)
		weak_form_kernel!(du, u, element, mesh,
			nonconservative_terms, equations,
			dg, cache, gpu = true) # add argument
	end

	return nothing
end

@inline function weak_form_kernel!(du, u,
	element, mesh::Union{TreeMesh{1}, StructuredMesh{1}},
	nonconservative_terms::False, equations,
	dg::DGSEM, cache, alpha = true, gpu = true) # add argument

	@unpack derivative_dhat = dg.basis

	for i in eachnode(dg)
		u_node = get_node_vars(u, equations, dg, i, element) # rewrite as kernel1

		flux1 = flux(u_node, 1, equations)
		for ii in eachnode(dg)
			multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], flux1, equations, dg, ii, element) # rewrite as kernel2
		end
	end

	#= # above nested for loops maybe like 
	threads = ##
	blocks = ##
	@cuda threads = threads, blocks = blocks apply_kernel1_kernel2(...)
	=#
	return nothing
end


#= What about the data transfer?
Should I ensure one round trip (CPU-GPU-CPU) for each function, even within inner functions such as get_node_vars()?
Or does it depend on the specific situation (I can decide using my judgment)?  =#