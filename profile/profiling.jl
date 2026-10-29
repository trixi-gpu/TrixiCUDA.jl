using Trixi, TrixiCUDA
using CUDA

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
                periodicity = true)

semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver,
                                           source_terms = source_terms_convergence_test, boundary_conditions = boundary_condition_periodic)

tspan_gpu = (0.0, 1.0)
t_gpu = 0.0

# Semi on GPU
equations_gpu, mesh_gpu, solver_gpu = semi_gpu.equations, semi_gpu.mesh, semi_gpu.solver
cache_gpu, cache_cpu = semi_gpu.cache_gpu, semi_gpu.cache_cpu
boundary_conditions_gpu, source_terms_gpu = semi_gpu.boundary_conditions, semi_gpu.source_terms

# ODE on GPU
ode_gpu = semidiscretizeGPU(semi_gpu, tspan_gpu)
u_gpu = copy(ode_gpu.u0)
du_gpu = similar(u_gpu)

# Volume integral test
TrixiCUDA.calc_volume_integral!(CUDA.CUDABackend(), du_gpu, u_gpu, mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu), equations_gpu,
                                solver_gpu.volume_integral, solver_gpu, cache_gpu, cache_cpu)

# Prolong to interfaces test
TrixiCUDA.prolong2interfaces!(CUDA.CUDABackend(), cache_gpu, u_gpu, mesh_gpu, equations_gpu, solver_gpu)

# Interface flux test
TrixiCUDA.calc_interface_flux!(CUDA.CUDABackend(), cache_gpu.elements.surface_flux_values, mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu),
                               equations_gpu, solver_gpu.surface_integral, solver_gpu, cache_gpu)

# Prolong to boundaries test
TrixiCUDA.prolong2boundaries!(CUDA.CUDABackend(), cache_gpu, u_gpu, mesh_gpu, equations_gpu, solver_gpu)

# Boundary flux test
TrixiCUDA.calc_boundary_flux!(CUDA.CUDABackend(), cache_gpu, t_gpu, boundary_conditions_gpu, mesh_gpu, equations_gpu, solver_gpu.surface_integral,
                              solver_gpu)

# Prolong to mortars test
TrixiCUDA.prolong2mortars!(cache_gpu, u_gpu, mesh_gpu, equations_gpu, solver_gpu.mortar, solver_gpu)

# Mortar flux test
TrixiCUDA.calc_mortar_flux!(cache_gpu.elements.surface_flux_values, mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu), equations_gpu,
                            solver_gpu.mortar, solver_gpu.surface_integral, solver_gpu, cache_gpu)

# Surface integral test
TrixiCUDA.calc_surface_integral!(CUDA.CUDABackend(), du_gpu, u_gpu, mesh_gpu, equations_gpu, solver_gpu.surface_integral, solver_gpu, cache_gpu)

# Jacobian test
TrixiCUDA.apply_jacobian!(CUDA.CUDABackend(), du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)

# Sources test
TrixiCUDA.calc_sources!(CUDA.CUDABackend(), du_gpu, u_gpu, t_gpu, source_terms_gpu, equations_gpu, solver_gpu, cache_gpu)
