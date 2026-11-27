# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_euler_mortar.jl
function (; polydeg = 3, initial_refinement_level = 2)
    return (;
            equations = CompressibleEulerEquations3D(1.4),
            initial_condition = initial_condition_convergence_test,

            coordinates_min = (0.0, 0.0, 0.0),
            coordinates_max = (2.0, 2.0, 2.0),
            initial_refinement_level,
            refinement_patches = ((type = "box", coordinates_min = (0.5, 0.5, 0.5),
                                   coordinates_max = (1.5, 1.5, 1.5)),),
            periodicity = true,

            polydeg,
            surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
            volume_integral = VolumeIntegralWeakForm(),

            boundary_conditions = boundary_condition_periodic,
            source_terms = source_terms_convergence_test,)
end
