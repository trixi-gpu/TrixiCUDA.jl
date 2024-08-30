using Trixi, TrixiGPU
using OrdinaryDiffEq
using CUDA
using Test

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
refinement_patches = ((type = "box", coordinates_min = (0.0, -1.0),
                       coordinates_max = (1.0, 1.0)),)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                refinement_patches = refinement_patches,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
(; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi

# Get deep copy for GPU to avoid overwriting during tests
mesh_gpu, equations_gpu = deepcopy(mesh), deepcopy(equations)
initial_condition_gpu, boundary_conditions_gpu, source_terms_gpu = deepcopy(initial_condition),
                                                                   deepcopy(boundary_conditions),
                                                                   deepcopy(source_terms)
solver_gpu, cache_gpu = deepcopy(solver), deepcopy(cache)

t = t_gpu = 0.0
tspan = (0.0, 1.0)

ode = semidiscretize(semi, tspan)
u_ode = copy(ode.u0)
du_ode = similar(u_ode)
u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

du_gpu, u_gpu = TrixiGPU.copy_to_device!(du, u)

Trixi.reset_du!(du, solver, cache)

# Test `cuda_volume_integral!`
TrixiGPU.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                               Trixi.have_nonconservative_terms(equations_gpu),
                               equations_gpu, solver_gpu.volume_integral, solver_gpu)
Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                            equations, solver.volume_integral, solver, cache)

# Test `cuda_prolong2interfaces!`
TrixiGPU.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)

# Test `cuda_interface_flux!`
TrixiGPU.cuda_interface_flux!(mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu),
                              equations_gpu, solver_gpu, cache_gpu)
Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                           Trixi.have_nonconservative_terms(equations), equations,
                           solver.surface_integral, solver, cache)

# Test `cuda_prolong2boundaries!`
TrixiGPU.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu, equations_gpu,
                                  cache_gpu)
Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)

# Test `cuda_boundary_flux!`
TrixiGPU.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu, equations_gpu,
                             solver_gpu, cache_gpu)
Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                          solver.surface_integral, solver)

# Test `cuda_prolong2mortars!`
TrixiGPU.cuda_prolong2mortars!(u_gpu, mesh_gpu, TrixiGPU.check_cache_mortars(cache_gpu), solver_gpu,
                               cache_gpu)
Trixi.prolong2mortars!(cache, u, mesh, equations,
                       solver.mortar, solver.surface_integral, solver)

TrixiGPU.cuda_mortar_flux!(mesh_gpu, TrixiGPU.check_cache_mortars(cache_gpu),
                           Trixi.have_nonconservative_terms(equations_gpu), equations_gpu,
                           solver_gpu, cache_gpu)
Trixi.calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                        Trixi.have_nonconservative_terms(equations), equations,
                        solver.mortar, solver.surface_integral, solver, cache)
@test cache_gpu.elements.surface_flux_values ≈ cache.elements.surface_flux_values
