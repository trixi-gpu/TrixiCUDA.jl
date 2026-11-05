() -> (;
       equations = IdealGlmMhdEquations1D(2.0),

       coordinates_min = (0.0,),
       coordinates_max = (1.0,),
       initial_refinement_level = 2,
       periodicity = true,

       polydeg = 3,
       surface_flux = flux_hindenlang_gassner,
       volume_integral = VolumeIntegralFluxDifferencing(flux_hindenlang_gassner),

       boundary_conditions = boundary_condition_periodic,
       source_terms = nothing,)
