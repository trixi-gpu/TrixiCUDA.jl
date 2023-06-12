include("header.jl") # Remove it after first run to avoid recompilation include("header.jl")

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

# Rewrite reset_du!()
#= function reset_du!(du, dg, cache)
	@threaded for element in eachelement(dg, cache)
		du[.., element] .= zero(eltype(du))
	end

	return du
end =#

function copy_to_gpu!(du, u, dg, cache)
	index = threadIdx().x
	stride = blockDim().x

	for i in index:stride:nelements(dg, cache)
		@inbounds du[.., i] = CuArray(du[.., i])
	end
end


threads = nelements(dg, cache) # nelements() from `trixi/src/solvers/dgsem_tree/containers_1d.jl`



############################################################
