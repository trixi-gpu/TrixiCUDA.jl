#= include("../cuda_dg_3d.jl") =#

# Run on CPU
#################################################################################
advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=3,
    n_cells_max=30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)

tspan = (0.0, 1.0)

ode_cpu = semidiscretize_cpu(semi, tspan)

sol_cpu = OrdinaryDiffEq.solve(ode_cpu, BS3(), adaptive=false, dt=0.01;
    abstol=1.0e-6, reltol=1.0e-6, ode_default_options()...)

# Run on GPU
#################################################################################
advection_velocity = (0.2f0, -0.7f0, 0.5f0)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0f0, -1.0f0, -1.0f0)
coordinates_max = (1.0f0, 1.0f0, 1.0f0)

mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=3,
    n_cells_max=30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)

tspan = (0.0f0, 1.0f0)

ode_gpu = semidiscretize_gpu(semi, tspan)

sol_gpu = OrdinaryDiffEq.solve(ode_gpu, BS3(), adaptive=false, dt=0.01;
    abstol=1.0e-6, reltol=1.0e-6, ode_default_options()...)

# Compare results
################################################################################
extrema(sol_cpu.u[end] - sol_gpu.u[end])