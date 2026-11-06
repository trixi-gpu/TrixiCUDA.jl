# https://github.com/trixi-framework/TrixiShallowWater.jl/blob/v0.3.3/examples/tree_1d_dgsem/elixir_shallowwater_ec.jl
function (; polydeg = 4, initial_refinement_level = 4)
    # Keep topography discontinuities aligned with cell interfaces when changing the mesh.
    function initial_condition_ec_discontinuous_bottom(x, t,
                                                       equations::ShallowWaterEquations1D)
        # Set the background values
        H = 4.25
        v = 0.0
        b = sin(x[1]) # arbitrary continuous function

        # Setup the discontinuous water height and velocity
        if x[1] >= 0.125 && x[1] <= 0.25
            H = 5.0
            v = 0.1882
        end

        # Setup a discontinuous bottom topography
        if x[1] >= -0.25 && x[1] <= -0.125
            b = 2.0 + 0.5 * sin(2.0 * pi * x[1])
        end

        return prim2cons(SVector(H, v, b), equations)
    end

    return (;
            equations = ShallowWaterEquations1D(gravity = 9.81),
            initial_condition = initial_condition_ec_discontinuous_bottom,

            coordinates_min = (-1.0,),
            coordinates_max = (1.0,),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
            volume_integral = VolumeIntegralFluxDifferencing((flux_wintermeyer_etal,
                                                              flux_nonconservative_wintermeyer_etal)),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
