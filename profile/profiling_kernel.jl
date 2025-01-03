using Trixi, TrixiCUDA

# Set the precision
RealT = Float32

# Set up the problem
advection_velocity = 1.0f0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs, RealT = RealT)

coordinates_min = -1.0f0
coordinates_max = 1.0f0

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000, RealT = RealT)

# Cache initialization
semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition_convergence_test,
                                           solver)

tspan_gpu = (0.0f0, 1.0f0)
t_gpu = 0.0f0

# Semi on GPU
equations_gpu = semi_gpu.equations
mesh_gpu, solver_gpu, cache_gpu = semi_gpu.mesh, semi_gpu.solver, semi_gpu.cache
boundary_conditions_gpu = semi_gpu.boundary_conditions
source_terms_gpu = semi_gpu.source_terms

# ODE on GPU
ode_gpu = semidiscretizeGPU(semi_gpu, tspan_gpu)
u_gpu = copy(ode_gpu.u0)
du_gpu = similar(u_gpu)

# Reset du and volume integral
TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                Trixi.have_nonconservative_terms(equations_gpu),
                                equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                cache_gpu)
