# The header part for testing false nonconservative terms in 1D
gamma = Float32(5 / 3)
equations = IdealGlmMhdEquations1D(gamma)

initial_condition = initial_condition_convergence_test

volume_flux = flux_hindenlang_gassner
solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs,
    volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = 0.0f0
coordinates_max = 1.0f0
mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=2,
    n_cells_max=10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

t = 0.0f0
tspan = (0.0f0, 2.0f0)

ode = semidiscretize(semi, tspan)
u_ode = copy(ode.u0)
du_ode = similar(u_ode)
u = wrap_array(u_ode, mesh, equations, solver, cache)
du = wrap_array(du_ode, mesh, equations, solver, cache)