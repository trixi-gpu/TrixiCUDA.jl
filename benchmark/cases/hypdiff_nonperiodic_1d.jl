# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_hypdiff_nonperiodic.jl
function (; polydeg = 4, initial_refinement_level = 3)
    return (;
            equations = HyperbolicDiffusionEquations1D(),
            initial_condition = initial_condition_poisson_nonperiodic,

            coordinates_min = (0.0,),
            coordinates_max = (1.0,),
            initial_refinement_level,
            periodicity = (false,),

            polydeg,
            surface_flux = flux_lax_friedrichs,
            volume_integral = VolumeIntegralWeakForm(),

            boundary_conditions = (; x_neg = boundary_condition_poisson_nonperiodic,
                                   x_pos = boundary_condition_poisson_nonperiodic),
            source_terms = source_terms_poisson_nonperiodic,)
end
