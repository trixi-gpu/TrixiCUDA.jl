() -> (;
       equations = ShallowWaterEquations1D(gravity = 9.81,
                                           threshold_partially_wet = 1.0e-4,
                                           threshold_desingularization = 1.0e-10),

       coordinates_min = (-1.0,),
       coordinates_max = (1.0,),
       initial_refinement_level = 2,
       periodicity = true,

       polydeg = 4,
       surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
       volume_integral = VolumeIntegralFluxDifferencing((flux_wintermeyer_etal,
                                                         flux_nonconservative_wintermeyer_etal)),

       boundary_conditions = boundary_condition_periodic,
       source_terms = nothing,)
