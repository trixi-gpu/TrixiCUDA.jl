include("test_trixicuda.jl")

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0
coordinates_max = 1.0

mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level = 4,
                n_cells_max = 30_000)

semi = TrixiCUDA.SemidiscretizationHyperbolic_gpu(mesh, equations,
                                                  initial_condition_convergence_test, solver)
