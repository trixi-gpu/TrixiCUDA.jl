using Trixi

equations = IdealGlmMhdMulticomponentEquations1D(gammas = (2.0, 2.0, 2.0),
                                                 gas_constants = (2.0, 2.0, 2.0))

initial_condition = initial_condition_weak_blast_wave

volume_flux = flux_hindenlang_gassner
solver = DGSEM(polydeg = 3, surface_flux = flux_hindenlang_gassner,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = 0.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

(; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi

surface_flux_values = cache.elements.surface_flux_values
n_boundaries_per_direction = cache.boundaries.n_boundaries_per_direction

# Calculate indices
lasts = accumulate(+, n_boundaries_per_direction)
firsts = lasts - n_boundaries_per_direction .+ 1
