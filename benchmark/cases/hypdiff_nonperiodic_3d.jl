# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.16/examples/tree_3d_dgsem/elixir_hypdiff_nonperiodic.jl
function (; polydeg = 4, initial_refinement_level = 2)
    return (;
            equations = HyperbolicDiffusionEquations3D(),
            initial_condition = initial_condition_poisson_nonperiodic,

            coordinates_min = (0.0, 0.0, 0.0),
            coordinates_max = (1.0, 1.0, 1.0),
            initial_refinement_level,
            periodicity = (false, true, true),

            polydeg,
            surface_flux = flux_lax_friedrichs,
            volume_integral = VolumeIntegralWeakForm(),

            boundary_conditions = (; x_neg = boundary_condition_poisson_nonperiodic,
                                   x_pos = boundary_condition_poisson_nonperiodic,
                                   y_neg = boundary_condition_periodic,
                                   y_pos = boundary_condition_periodic,
                                   z_neg = boundary_condition_periodic,
                                   z_pos = boundary_condition_periodic),
            source_terms = source_terms_poisson_nonperiodic,)
end
