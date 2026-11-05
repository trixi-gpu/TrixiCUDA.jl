() -> (;
       equations = IdealGlmMhdEquations2D(5.0 / 3; initial_c_h = 1.0),

       coordinates_min = (0.0, 0.0),
       coordinates_max = (sqrt(2.0), sqrt(2.0)),
       initial_refinement_level = 2,
       refinement_patches = ((type = "box", coordinates_min = 0.25 .* (sqrt(2.0), sqrt(2.0)),
                              coordinates_max = 0.75 .* (sqrt(2.0), sqrt(2.0))),),
       periodicity = true,

       polydeg = 3,
       surface_flux = (flux_hlle, flux_nonconservative_powell),
       volume_integral = VolumeIntegralFluxDifferencing((flux_hindenlang_gassner, flux_nonconservative_powell)),

       boundary_conditions = boundary_condition_periodic,
       source_terms = nothing,)
