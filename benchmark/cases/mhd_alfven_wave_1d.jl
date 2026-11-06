# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.16/examples/tree_1d_dgsem/elixir_mhd_alfven_wave.jl
function (; polydeg = 4, initial_refinement_level = 2)
    return (;
            equations = IdealGlmMhdEquations1D(5.0 / 3),
            initial_condition = initial_condition_convergence_test,

            coordinates_min = (0.0,),
            coordinates_max = (1.0,),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
            volume_integral = VolumeIntegralFluxDifferencing(flux_hindenlang_gassner),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
