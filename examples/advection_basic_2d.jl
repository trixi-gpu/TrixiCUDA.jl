# 2D Linear Advection Equation
using Trixi, TrixiGPU
using OrdinaryDiffEq

advection_velocity = (0.2f0, -0.7f0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0f0, -1.0f0)
coordinates_max = (1.0f0, 1.0f0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

tspan = (0.0f0, 1.0f0)

# FIXME: Remember to export
ode = TrixiGPU.semidiscretize_gpu(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100)

save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.6)

callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, save_everystep = false, callback = callbacks);
summary_callback()
