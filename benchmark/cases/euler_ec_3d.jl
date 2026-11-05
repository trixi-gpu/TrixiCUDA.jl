() -> (;
       equations = CompressibleEulerEquations3D(1.4),

       coordinates_min = (-2.0, -2.0, -2.0),
       coordinates_max = (2.0, 2.0, 2.0),
       initial_refinement_level = 3,
       periodicity = true,

       polydeg = 3,
       surface_flux = flux_ranocha,
       volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),

       boundary_conditions = boundary_condition_periodic,
       source_terms = nothing,)
