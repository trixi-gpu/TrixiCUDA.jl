#= include("../cuda_dg_2d.jl") =#

# Run on CPU
#################################################################################
advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
refinement_patches = (
    (type="box", coordinates_min=(0.0, -1.0), coordinates_max=(1.0, 1.0)),
)
mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=2,
    refinement_patches=refinement_patches,
    n_cells_max=10_000,)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 1.0)

ode_cpu = semidiscretize_cpu(semi, tspan)

#= sol_cpu = OrdinaryDiffEq.solve(ode_cpu, BS3(), adaptive=false, dt=0.01;
    abstol=1.0e-6, reltol=1.0e-6, ode_default_options()...) =#

#= u0_ode_cpu = copy(ode_cpu.u0)
du_ode_cpu = similar(u0_ode_cpu)
Trixi.rhs!(du_ode_cpu, u0_ode_cpu, semi, 0.0) =#

# Run on GPU
#################################################################################
advection_velocity = (0.2f0, -0.7f0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0f0, -1.0f0)
coordinates_max = (1.0f0, 1.0f0)
refinement_patches = (
    (type="box", coordinates_min=(0.0f0, -1.0f0), coordinates_max=(1.0f0, 1.0f0)),
)
mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=2,
    refinement_patches=refinement_patches,
    n_cells_max=10_000,)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0f0, 1.0f0)

ode_gpu = semidiscretize_gpu(semi, tspan)

#= sol_gpu = OrdinaryDiffEq.solve(ode_gpu, BS3(), adaptive=false, dt=0.01;
    abstol=1.0e-6, reltol=1.0e-6, ode_default_options()...) =#

#= u0_ode_gpu = copy(ode_gpu.u0)
du_ode_gpu = similar(u0_ode_gpu)
Trixi.rhs!(du_ode_gpu, u0_ode_gpu, semi, 0.0f0) =#

# Compare results
################################################################################
#= extrema(sol_cpu.u[end] - sol_gpu.u[end]) =#

# Step inspection
################################################################################
integrator_cpu = init(ode_cpu, BS3())
integrator_gpu = init(ode_gpu, BS3())

global step = 0
while (integrator_cpu.t < tspan[2] && integrator_gpu.t < tspan[2])
    step!(integrator_cpu)
    step!(integrator_gpu)

    global step += 1

    if (maximum(abs.(integrator_cpu.u - integrator_gpu.u))) > 0.1
        println("Step:", step)
        println("CPU:", integrator_cpu.u, " ", integrator_cpu.t, " ", integrator_cpu.dt)
        println("GPU:", integrator_gpu.u, " ", integrator_gpu.t, " ", integrator_gpu.dt)
    end
end