using Trixi, OrdinaryDiffEq

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

ode = semidiscretize_new(semi, (0.0, 1.0))

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0; abstol=1.0e-6, reltol=1.0e-6, ode_default_options()...)
