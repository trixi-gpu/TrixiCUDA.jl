using Trixi, TrixiGPU
using OrdinaryDiffEq
using CUDA

equations = HyperbolicDiffusionEquations3D()

initial_condition = initial_condition_poisson_nonperiodic
boundary_conditions = (x_neg = boundary_condition_poisson_nonperiodic,
                       x_pos = boundary_condition_poisson_nonperiodic,
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic,
                       z_neg = boundary_condition_periodic,
                       z_pos = boundary_condition_periodic)

solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

coordinates_min = (0.0, 0.0, 0.0)
coordinates_max = (1.0, 1.0, 1.0)
mesh = TreeMesh(coordinates_min,
                coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 30_000,
                periodicity = (false, true, true))

semi = SemidiscretizationHyperbolic(mesh,
                                    equations,
                                    initial_condition,
                                    solver,
                                    source_terms = source_terms_poisson_nonperiodic,
                                    boundary_conditions = boundary_conditions)

(; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi

t = 0.0
tspan = (0.0, 5.0)

ode = semidiscretize(semi, tspan)
u_ode = copy(ode.u0)
du_ode = similar(u_ode)
u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

du, u = TrixiGPU.copy_to_device!(du, u)
TrixiGPU.cuda_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations), equations,
                               solver.volume_integral, solver)
TrixiGPU.cuda_prolong2interfaces!(u, mesh, equations, cache)
TrixiGPU.cuda_interface_flux!(mesh, Trixi.have_nonconservative_terms(equations), equations, solver,
                              cache)
TrixiGPU.cuda_prolong2boundaries!(u, mesh, boundary_conditions, equations, cache)

TrixiGPU.cuda_boundary_flux!(t, mesh, boundary_conditions, equations, solver, cache)
TrixiGPU.cuda_surface_integral!(du, mesh, equations, solver, cache)
TrixiGPU.cuda_jacobian!(du, mesh, equations, cache)
TrixiGPU.cuda_sources!(du, u, t, source_terms, equations, cache)
