using Trixi, TrixiCUDA
using OrdinaryDiffEq

# Currently skip the issue of scalar indexing
using CUDA
CUDA.allowscalar(true)

# The example is taken from the Trixi.jl

###############################################################################
# semidiscretization of the hyperbolic diffusion equations

equations = HyperbolicDiffusionEquations2D()

initial_condition = initial_condition_poisson_nonperiodic

boundary_conditions = (x_neg = boundary_condition_poisson_nonperiodic,
                       x_pos = boundary_condition_poisson_nonperiodic,
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic)

solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000,
                periodicity = (false, true))

semi = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver,
                                       boundary_conditions = boundary_conditions,
                                       source_terms = source_terms_poisson_nonperiodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretizeGPU(semi, tspan) # from TrixiCUDA.jl

summary_callback = SummaryCallback()

resid_tol = 5.0e-12
steady_state_callback = SteadyStateCallback(abstol = resid_tol, reltol = 0.0)

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback, steady_state_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks)
summary_callback() # print the timer summary
