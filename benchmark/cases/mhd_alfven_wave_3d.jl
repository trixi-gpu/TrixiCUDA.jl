() -> (;
       equations = IdealGlmMhdEquations3D(5.0 / 3; initial_c_h = 1.0),

       coordinates_min = (-1.0, -1.0, -1.0),
       coordinates_max = (1.0, 1.0, 1.0),
       initial_refinement_level = 2,
       periodicity = true,

       polydeg = 3,
       surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
       volume_integral = VolumeIntegralFluxDifferencing((flux_hindenlang_gassner, flux_nonconservative_powell)),

       boundary_conditions = boundary_condition_periodic,
       source_terms = nothing,)
