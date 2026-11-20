# https://github.com/trixi-framework/TrixiShallowWater.jl/blob/v0.3.3/examples/tree_1d_dgsem/elixir_shallowwater_shock_capturing.jl
function (; polydeg = 4, initial_refinement_level = 3)
    # Stone throw with a discontinuous bottom topography
    function initial_condition_stone_throw_discontinuous_bottom(x, t,
                                                                equations::ShallowWaterEquations1D)
        H = equations.H0
        v = 0.0
        if x[1] >= -0.75 && x[1] <= 0.0
            v = -1.0
        elseif x[1] >= 0.0 && x[1] <= 0.75
            v = 1.0
        end

        b = (1.5 / exp(0.5 * ((x[1] - 1.0)^2)) +
             0.75 / exp(0.5 * ((x[1] + 1.0)^2)))
        if x[1] >= -1.5 && x[1] <= 0.0
            b = 0.5
        end

        return prim2cons(SVector(H, v, b), equations)
    end

    equations = ShallowWaterEquations1D(gravity = 9.812, H0 = 1.75)
    basis = LobattoLegendreBasis(Float64, polydeg)
    indicator = IndicatorHennemannGassner(equations, basis;
                                          alpha_max = 0.5, alpha_min = 0.001,
                                          alpha_smooth = true, variable = waterheight_pressure)
    surface_flux = (FluxHydrostaticReconstruction(FluxLaxFriedrichs(max_abs_speed_naive),
                                                  hydrostatic_reconstruction_audusse_etal),
                    flux_nonconservative_audusse_etal)
    return (;
            equations,
            initial_condition = initial_condition_stone_throw_discontinuous_bottom,

            coordinates_min = (-3.0,),
            coordinates_max = (3.0,),
            initial_refinement_level,
            periodicity = (false,),

            polydeg,
            surface_flux,
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = (flux_wintermeyer_etal,
                                                                               flux_nonconservative_wintermeyer_etal),
                                                             volume_flux_fv = surface_flux),

            boundary_conditions = (; x_neg = boundary_condition_slip_wall,
                                   x_pos = boundary_condition_slip_wall),
            source_terms = nothing,)
end
