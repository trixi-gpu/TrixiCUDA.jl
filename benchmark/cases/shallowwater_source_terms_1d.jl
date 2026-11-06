# https://github.com/trixi-framework/TrixiShallowWater.jl/blob/v0.3.3/examples/tree_1d_dgsem/elixir_shallowwater_source_terms.jl
function (; polydeg = 3, initial_refinement_level = 3)
    return (;
            equations = ShallowWaterEquations1D(gravity = 9.81),
            initial_condition = initial_condition_convergence_test,

            coordinates_min = (0.0,),
            coordinates_max = (sqrt(2.0),),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_fjordholm_etal),
            volume_integral = VolumeIntegralFluxDifferencing((flux_wintermeyer_etal,
                                                              flux_nonconservative_wintermeyer_etal)),

            boundary_conditions = boundary_condition_periodic,
            source_terms = source_terms_convergence_test,)
end
