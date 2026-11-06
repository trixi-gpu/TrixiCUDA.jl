# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.16/examples/tree_2d_dgsem/elixir_mhd_alfven_wave.jl
# Fix the GLM speed explicitly; RHS benchmarks do not run GlmSpeedCallback.
function (; polydeg = 3, initial_refinement_level = 4)
    return (;
            equations = IdealGlmMhdEquations2D(5.0 / 3; initial_c_h = 1.0),
            initial_condition = initial_condition_convergence_test,

            coordinates_min = (0.0, 0.0),
            coordinates_max = (sqrt(2.0), sqrt(2.0)),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_powell),
            volume_integral = VolumeIntegralFluxDifferencing((flux_central, flux_nonconservative_powell)),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
