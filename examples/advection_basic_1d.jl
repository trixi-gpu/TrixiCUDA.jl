using Trixi, TrixiGPU
using OrdinaryDiffEq

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEM(; polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = -1.0
coordinates_max = 1.0

mesh = TreeMesh(
    coordinates_min, coordinates_max; initial_refinement_level=4, n_cells_max=30_000
)

semi = SemidiscretizationHyperbolic(
    mesh, equations, initial_condition_convergence_test, solver
)

tspan = (0.0, 1.0)

ode = semidiscretize_gpu(semi, tspan) # from TrixiGPU.jl

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(
    semi;
    interval=analysis_interval,
    extra_analysis_errors=(:l2_error_primitive, :linf_error_primitive),
)

alive_callback = AliveCallback(; analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(;
    interval=100,
    save_initial_solution=true,
    save_final_solution=true,
    solution_variables=cons2prim,
)

stepsize_callback = StepsizeCallback(; cfl=0.8)

callbacks = CallbackSet(
    summary_callback, analysis_callback, alive_callback, save_solution, stepsize_callback
)

sol = solve(
    ode,
    CarpenterKennedy2N54(; williamson_condition=false);
    dt=1.0,
    save_everystep=false,
    callback=callbacks,
);
summary_callback()
