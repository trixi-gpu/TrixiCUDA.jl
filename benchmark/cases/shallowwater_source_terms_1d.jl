() -> (;
       equations = ShallowWaterEquations1D(gravity = 9.81,
                                           threshold_partially_wet = 1.0e-4,
                                           threshold_desingularization = 1.0e-10),

       coordinates_min = (0.0,),
       coordinates_max = (sqrt(2.0),),
       initial_refinement_level = 4,
       periodicity = true,

       polydeg = 3,
       surface_flux = (flux_lax_friedrichs, flux_nonconservative_fjordholm_etal),
       volume_integral = VolumeIntegralFluxDifferencing((flux_wintermeyer_etal,
                                                         flux_nonconservative_wintermeyer_etal)),

       boundary_conditions = boundary_condition_periodic,
       source_terms = source_terms_convergence_test,)
