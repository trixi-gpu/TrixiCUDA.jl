using Trixi, TrixiCUDA
using CUDA
using BenchmarkTools

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_convergence_test
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (0.0, 0.0, 0.0)
coordinates_max = (2.0, 2.0, 2.0)
refinement_patches = ((type = "box", coordinates_min = (0.5, 0.5, 0.5),
                       coordinates_max = (1.5, 1.5, 1.5)),)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                refinement_patches = refinement_patches,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)
semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver,
                                           source_terms = source_terms_convergence_test)

tspan = tspan_gpu = (0.0, 1.0)
t = t_gpu = 0.0

# Semi on CPU
(; mesh, equations, boundary_conditions, source_terms, solver, cache) = semi

# Semi on GPU
equations_gpu = semi_gpu.equations
mesh_gpu, solver_gpu, cache_gpu = semi_gpu.mesh, semi_gpu.solver, semi_gpu.cache
boundary_conditions_gpu = semi_gpu.boundary_conditions
source_terms_gpu = semi_gpu.source_terms

# ODE on CPU
ode = semidiscretize(semi, tspan)
u_ode = copy(ode.u0)
du_ode = similar(u_ode)
u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

# ODE on GPU
ode_gpu = semidiscretizeGPU(semi_gpu, tspan_gpu)
u_gpu = copy(ode_gpu.u0)
du_gpu = similar(u_gpu)

# Semidiscretization process
# Reset du test
TrixiCUDA.reset_du!(du_gpu)

# Volume integral test
TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                Trixi.have_nonconservative_terms(equations_gpu),
                                equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                cache_gpu)

# Prolong to interfaces test
TrixiCUDA.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)

# Interface flux test
TrixiCUDA.cuda_interface_flux!(mesh_gpu,
                               Trixi.have_nonconservative_terms(equations_gpu),
                               equations_gpu, solver_gpu, cache_gpu)

# Prolong to boundaries test
TrixiCUDA.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu,
                                   equations_gpu, cache_gpu)

# Boundary flux test
TrixiCUDA.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu,
                              Trixi.have_nonconservative_terms(equations_gpu),
                              equations_gpu, solver_gpu, cache_gpu)

# Prolong to mortars test
@benchmark CUDA.@sync TrixiCUDA.cuda_prolong2mortars!(u_gpu, mesh_gpu,
                                                      TrixiCUDA.check_cache_mortars(cache_gpu),
                                                      solver_gpu, cache_gpu)

# # Mortar flux test
# @benchmark CUDA.@sync TrixiCUDA.cuda_mortar_flux!(mesh_gpu, TrixiCUDA.check_cache_mortars(cache_gpu),
#                             Trixi.have_nonconservative_terms(equations_gpu),
#                             equations_gpu, solver_gpu, cache_gpu)
