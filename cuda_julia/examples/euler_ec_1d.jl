#= include("../cuda_dg_1d.jl") =#

# Run on CPU
#################################################################################
equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_weak_blast_wave

volume_flux = flux_ranocha
solver = DGSEM(
    polydeg = 3,
    surface_flux = flux_ranocha,
    volume_integral = VolumeIntegralFluxDifferencing(volume_flux),
)

coordinates_min = (-2.0,)
coordinates_max = (2.0,)

mesh = TreeMesh(
    coordinates_min,
    coordinates_max,
    initial_refinement_level = 5,
    n_cells_max = 10_000,
)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 0.4)

ode_cpu = semidiscretize_cpu(semi, tspan)

sol_cpu = OrdinaryDiffEq.solve(
    ode_cpu,
    BS3(); #= SSPRK43() =#
    abstol = 1.0e-6,
    reltol = 1.0e-6,
    ode_default_options()...,
)

# Run on GPU
#################################################################################
equations = CompressibleEulerEquations1D(1.4f0)

initial_condition = initial_condition_weak_blast_wave

volume_flux = flux_ranocha
solver = DGSEM(
    polydeg = 3,
    surface_flux = flux_ranocha,
    volume_integral = VolumeIntegralFluxDifferencing(volume_flux),
)

coordinates_min = (-2.0f0,)
coordinates_max = (2.0f0,)

mesh = TreeMesh(
    coordinates_min,
    coordinates_max,
    initial_refinement_level = 5,
    n_cells_max = 10_000,
)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0f0, 0.4f0)

ode_gpu = semidiscretize_gpu(semi, tspan)

sol_gpu = OrdinaryDiffEq.solve(
    ode_gpu,
    BS3(); #= SSPRK43() =#
    abstol = 1.0e-6,
    reltol = 1.0e-6,
    ode_default_options()...,
)

# Compare results
################################################################################
extrema(sol_cpu.u[end] - sol_gpu.u[end])
