# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_mhd_alfven_wave_mortar.jl
# Fix the GLM speed explicitly; RHS benchmarks do not run GlmSpeedCallback.
function (; polydeg = 3, initial_refinement_level = 4)
    return (;
            equations = IdealGlmMhdEquations2D(5.0 / 3; initial_c_h = 1.0),
            initial_condition = initial_condition_convergence_test,

            coordinates_min = (0.0, 0.0),
            coordinates_max = (sqrt(2.0), sqrt(2.0)),
            initial_refinement_level,
            refinement_patches = ((type = "box", coordinates_min = 0.25 .* (sqrt(2.0), sqrt(2.0)),
                                   coordinates_max = 0.75 .* (sqrt(2.0), sqrt(2.0))),),
            periodicity = true,

            polydeg,
            surface_flux = (flux_hlle, flux_nonconservative_powell),
            volume_integral = VolumeIntegralFluxDifferencing((flux_hindenlang_gassner,
                                                              flux_nonconservative_powell)),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
