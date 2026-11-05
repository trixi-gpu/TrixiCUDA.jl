() -> (;
       equations = HyperbolicDiffusionEquations1D(nu = 1.0, Lr = inv(2 * Float64(pi))),

       coordinates_min = (0.0,),
       coordinates_max = (1.0,),
       initial_refinement_level = 3,
       periodicity = (false,),

       polydeg = 4,
       surface_flux = flux_lax_friedrichs,
       volume_integral = VolumeIntegralWeakForm(),

       boundary_conditions = (; x_neg = boundary_condition_poisson_nonperiodic,
                              x_pos = boundary_condition_poisson_nonperiodic),
       source_terms = source_terms_poisson_nonperiodic,)
