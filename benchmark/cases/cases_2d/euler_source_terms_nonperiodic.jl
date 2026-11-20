# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_euler_source_terms_nonperiodic.jl
function (; polydeg = 3, initial_refinement_level = 4)
    boundary_condition = BoundaryConditionDirichlet(initial_condition_convergence_test)
    return (;
            equations = CompressibleEulerEquations2D(1.4),
            initial_condition = initial_condition_convergence_test,

            coordinates_min = (0.0, 0.0),
            coordinates_max = (2.0, 2.0),
            initial_refinement_level,
            periodicity = (false, false),

            polydeg,
            surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
            volume_integral = VolumeIntegralWeakForm(),

            boundary_conditions = (; x_neg = boundary_condition, x_pos = boundary_condition,
                                   y_neg = boundary_condition, y_pos = boundary_condition),
            source_terms = source_terms_convergence_test,)
end
