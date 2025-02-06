using Trixi, TrixiCUDA
using Test
using CUDA
CUDA.allowscalar(true)
include("./test/test_macros.jl")

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

initial_condition = initial_condition_convergence_test
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
solver_gpu = DGSEMGPU(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)
refinement_patches = ((type = "box", coordinates_min = (0.0, -1.0, -1.0),
                       coordinates_max = (1.0, 1.0, 1.0)),
                      (type = "box", coordinates_min = (0.0, -0.5, -0.5),
                       coordinates_max = (0.5, 0.5, 0.5)))
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                refinement_patches = refinement_patches,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver_gpu)

tspan = tspan_gpu = (0.0, 5.0)
t = t_gpu = 0.0

# Semi on CPU
(; mesh, equations, boundary_conditions, source_terms, solver, cache) = semi

# Semi on GPU
equations_gpu, mesh_gpu, solver_gpu = semi_gpu.equations, semi_gpu.mesh, semi_gpu.solver
cache_gpu, cache_cpu = semi_gpu.cache_gpu, semi_gpu.cache_cpu
boundary_conditions_gpu, source_terms_gpu = semi_gpu.boundary_conditions, semi_gpu.source_terms

# ODE on CPU
ode = semidiscretize(semi, tspan)
u_ode = copy(ode.u0)
du_ode = similar(u_ode)
u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

# ODE on GPU
ode_gpu = semidiscretizeGPU(semi_gpu, tspan_gpu)
u_gpu_ = copy(ode_gpu.u0)
du_gpu_ = similar(u_gpu_)
u_gpu = TrixiCUDA.wrap_array(u_gpu_, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
du_gpu = TrixiCUDA.wrap_array(du_gpu_, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)

# Tests for semidiscretization process
@test_approx (u_gpu, u) # du is initlaizaed as undefined, cannot test now
Trixi.reset_du!(du, solver, cache)

TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                Trixi.have_nonconservative_terms(equations_gpu),
                                equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                cache_gpu, cache_cpu)
Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                            equations, solver.volume_integral, solver, cache)
@test_approx (du_gpu, du)

TrixiCUDA.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
@test_approx (cache_gpu.interfaces.u, cache.interfaces.u)

TrixiCUDA.cuda_interface_flux!(mesh_gpu,
                               Trixi.have_nonconservative_terms(equations_gpu),
                               equations_gpu, solver_gpu, cache_gpu)
Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                           Trixi.have_nonconservative_terms(equations), equations,
                           solver.surface_integral, solver, cache)
@test_approx (cache_gpu.elements.surface_flux_values,
              cache.elements.surface_flux_values)

TrixiCUDA.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu,
                                   equations_gpu, cache_gpu)
Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
@test_approx (cache_gpu.boundaries.u, cache.boundaries.u)

TrixiCUDA.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu,
                              Trixi.have_nonconservative_terms(equations_gpu),
                              equations_gpu, solver_gpu, cache_gpu)
Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                          solver.surface_integral, solver)
@test_approx (cache_gpu.elements.surface_flux_values,
              cache.elements.surface_flux_values)

TrixiCUDA.cuda_prolong2mortars!(u_gpu, mesh_gpu,
                                TrixiCUDA.check_cache_mortars(cache_gpu),
                                solver_gpu, cache_gpu)
Trixi.prolong2mortars!(cache, u, mesh, equations, solver.mortar, solver)
diff = Array(cache_gpu.mortars.u_upper_left) .- cache.mortars.u_upper_left
println("DIFF:", diff)
println("MAX DIFF:", maximum(abs, diff))

# neighbor_ids = cache_gpu.mortars.neighbor_ids
# large_sides = cache_gpu.mortars.large_sides
# orientations = cache_gpu.mortars.orientations

# # The original CPU arrays hold NaNs
# u_upper_left = cache_gpu.mortars.u_upper_left
# u_upper_right = cache_gpu.mortars.u_upper_right
# u_lower_left = cache_gpu.mortars.u_lower_left
# u_lower_right = cache_gpu.mortars.u_lower_right
# forward_upper = solver_gpu.mortar.forward_upper
# forward_lower = solver_gpu.mortar.forward_lower

# prolong_mortars_small2small_kernel = @cuda launch=false TrixiCUDA.prolong_mortars_small2small_kernel!(u_upper_left,
#                                                                                                       u_upper_right,
#                                                                                                       u_lower_left,
#                                                                                                       u_lower_right,
#                                                                                                       u_gpu,
#                                                                                                       neighbor_ids,
#                                                                                                       large_sides,
#                                                                                                       orientations)
# prolong_mortars_small2small_kernel(u_upper_left, u_upper_right, u_lower_left, u_lower_right, u_gpu,
#                                    neighbor_ids, large_sides, orientations;
#                                    TrixiCUDA.kernel_configurator_3d(prolong_mortars_small2small_kernel,
#                                                                     size(u_upper_left, 2),
#                                                                     size(u_upper_left, 3)^2,
#                                                                     size(u_upper_left, 5))...)

# tmp_upper_left, tmp_upper_right = CUDA.zeros(eltype(u_upper_left), size(u_upper_left))

# prolong_mortars_large2small_kernel = @cuda launch=false TrixiCUDA.prolong_mortars_large2small_kernel!(u_upper_left,
#                                                                                                       u_upper_right,
#                                                                                                       u_lower_left,
#                                                                                                       u_lower_right,
#                                                                                                       tmp_upper_left,
#                                                                                                       tmp_upper_right,
#                                                                                                       tmp_lower_left,
#                                                                                                       tmp_lower_right,
#                                                                                                       u_gpu, forward_upper,
#                                                                                                       forward_lower,
#                                                                                                       neighbor_ids,
#                                                                                                       large_sides,
#                                                                                                       orientations)
# prolong_mortars_large2small_kernel(u_upper_left, u_upper_right, u_lower_left, u_lower_right,
#                                    tmp_upper_left, tmp_upper_right, tmp_lower_left,
#                                    tmp_lower_right, u_gpu, forward_upper, forward_lower, neighbor_ids,
#                                    large_sides, orientations; cooperative = true,
#                                    TrixiCUDA.kernel_configurator_coop_3d(prolong_mortars_large2small_kernel,
#                                                                          size(u_upper_left, 2),
#                                                                          size(u_upper_left, 3)^2,
#                                                                          size(u_upper_left, 5))...)

# tmp_upper_left1 = zero(similar(u_upper_left)) # undef to zero
# tmp_upper_right1 = zero(similar(u_upper_right)) # undef to zero
# tmp_lower_left1 = zero(similar(u_lower_left)) # undef to zero
# tmp_lower_right1 = zero(similar(u_lower_right)) # undef to zero

# prolong_mortars_large2small_kernel = @cuda launch=false TrixiCUDA.prolong_mortars_large2small_kernel!(tmp_upper_left1,
#                                                                                                       tmp_upper_right1,
#                                                                                                       tmp_lower_left1,
#                                                                                                       tmp_lower_right1,
#                                                                                                       u_gpu,
#                                                                                                       forward_upper,
#                                                                                                       forward_lower,
#                                                                                                       neighbor_ids,
#                                                                                                       large_sides,
#                                                                                                       orientations)
# prolong_mortars_large2small_kernel(tmp_upper_left1, tmp_upper_right1, tmp_lower_left1,
#                                    tmp_lower_right1, u_gpu, forward_upper, forward_lower,
#                                    neighbor_ids, large_sides, orientations;
#                                    TrixiCUDA.kernel_configurator_3d(prolong_mortars_large2small_kernel,
#                                                                     size(u_upper_left, 2),
#                                                                     size(u_upper_left, 3)^2,
#                                                                     size(u_upper_left, 5))...)
