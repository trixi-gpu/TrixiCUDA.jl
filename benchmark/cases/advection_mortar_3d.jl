# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_advection_mortar.jl
function (; polydeg = 3, initial_refinement_level = 2)
    return (;
            equations = LinearScalarAdvectionEquation3D((0.2, -0.7, 0.5)),
            initial_condition = initial_condition_convergence_test,

            coordinates_min = (-1.0, -1.0, -1.0),
            coordinates_max = (1.0, 1.0, 1.0),
            initial_refinement_level,
            refinement_patches = ((type = "box", coordinates_min = (0.0, -1.0, -1.0),
                                   coordinates_max = (1.0, 1.0, 1.0)),
                                  (type = "box", coordinates_min = (0.0, -0.5, -0.5),
                                   coordinates_max = (0.5, 0.5, 0.5))),
            periodicity = true,

            polydeg,
            surface_flux = flux_lax_friedrichs,
            volume_integral = VolumeIntegralWeakForm(),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
