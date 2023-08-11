# The header part for testing true nonconservative terms in 2D
gamma = Float32(5 / 3)
equations = IdealGlmMhdEquations2D(gamma)

initial_condition = initial_condition_convergence_test

volume_flux = (flux_central, flux_nonconservative_powell)
solver = DGSEM(polydeg=3, surface_flux=(flux_lax_friedrichs, flux_nonconservative_powell),
    volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0.0f0, 0.0f0)
coordinates_max = (Float32(sqrt(2.0)), Float32(sqrt(2.0)))
mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=4,
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