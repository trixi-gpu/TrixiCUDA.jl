# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_mhd_rotor.jl
# Fix the GLM speed explicitly; RHS benchmarks do not run GlmSpeedCallback.
function (; polydeg = 4, initial_refinement_level = 4)
    # Classical MHD rotor, setup of Derigs et al. (2018); domain [0, 1]^2 with gamma = 1.4
    function initial_condition_rotor(x, t, equations::IdealGlmMhdEquations2D)
        dx = x[1] - 0.5
        dy = x[2] - 0.5
        r = sqrt(dx^2 + dy^2)
        f = (0.115 - r) / 0.015
        if r <= 0.1
            rho = 10.0
            v1 = -20 * dy
            v2 = 20 * dx
        elseif r >= 0.115
            rho = 1.0
            v1 = 0.0
            v2 = 0.0
        else
            rho = 1 + 9 * f
            v1 = -20 * f * dy
            v2 = 20 * f * dx
        end
        v3 = 0.0
        p = 1.0
        B1 = 5 / sqrt(4 * pi)
        B2 = 0.0
        B3 = 0.0
        psi = 0.0
        return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
    end

    equations = IdealGlmMhdEquations2D(1.4; initial_c_h = 1.0)
    basis = LobattoLegendreBasis(Float64, polydeg)
    indicator = IndicatorHennemannGassner(equations, basis;
                                          alpha_max = 0.5, alpha_min = 0.001,
                                          alpha_smooth = true, variable = density_pressure)
    surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_powell)
    return (;
            equations,
            initial_condition = initial_condition_rotor,

            coordinates_min = (0.0, 0.0),
            coordinates_max = (1.0, 1.0),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux,
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = (flux_hindenlang_gassner,
                                                                               flux_nonconservative_powell),
                                                             volume_flux_fv = surface_flux),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
