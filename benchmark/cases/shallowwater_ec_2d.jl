# https://github.com/trixi-framework/TrixiShallowWater.jl/blob/v0.3.3/examples/tree_2d_dgsem/elixir_shallowwater_ec.jl
function (; polydeg = 4, initial_refinement_level = 2)
    # This upstream EC state uses element IDs and requires the default level-2 mesh.
    function initial_condition_ec_discontinuous_bottom(x, t, element_id,
                                                       equations::ShallowWaterEquations2D)
        # Set up polar coordinates
        inicenter = SVector(0.7, 0.7)
        x_norm = x[1] - inicenter[1]
        y_norm = x[2] - inicenter[2]
        r = sqrt(x_norm^2 + y_norm^2)
        phi = atan(y_norm, x_norm)
        sin_phi, cos_phi = sincos(phi)

        # Set the background values
        H = 4.25
        v1 = 0.0
        v2 = 0.0
        b = 0.0

        # setup the discontinuous water height and velocities
        if element_id == 10
            H = 5.0
            v1 = 0.1882 * cos_phi
            v2 = 0.1882 * sin_phi
        end

        # Setup a discontinuous bottom topography using the element id number
        if element_id == 7
            b = 2.0 + 0.5 * sin(2.0 * pi * x[1]) + 0.5 * cos(2.0 * pi * x[2])
        end

        return prim2cons(SVector(H, v1, v2, b), equations)
    end

    return (;
            equations = ShallowWaterEquations2D(gravity = 9.81),
            initial_condition = initial_condition_ec_discontinuous_bottom,

            coordinates_min = (-1.0, -1.0),
            coordinates_max = (1.0, 1.0),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
            volume_integral = VolumeIntegralFluxDifferencing((flux_wintermeyer_etal,
                                                              flux_nonconservative_wintermeyer_etal)),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
