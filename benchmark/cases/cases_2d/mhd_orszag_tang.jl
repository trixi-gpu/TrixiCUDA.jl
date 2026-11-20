# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_mhd_orszag_tang.jl
# Fix the GLM speed explicitly; RHS benchmarks do not run GlmSpeedCallback.
function (; polydeg = 3, initial_refinement_level = 4)
    # Classical Orszag-Tang vortex, setup of Derigs et al. (2018)
    function initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations2D)
        rho = 1.0
        v1 = -sinpi(2 * x[2])
        v2 = sinpi(2 * x[1])
        v3 = 0.0
        p = 1 / equations.gamma
        B1 = -sinpi(2 * x[2]) / equations.gamma
        B2 = sinpi(4 * x[1]) / equations.gamma
        B3 = 0.0
        psi = 0.0
        return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
    end

    equations = IdealGlmMhdEquations2D(5.0 / 3; initial_c_h = 1.0)
    basis = LobattoLegendreBasis(Float64, polydeg)
    indicator = IndicatorHennemannGassner(equations, basis;
                                          alpha_max = 0.5, alpha_min = 0.001,
                                          alpha_smooth = true, variable = density_pressure)
    surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_powell)
    return (;
            equations,
            initial_condition = initial_condition_orszag_tang,

            coordinates_min = (0.0, 0.0),
            coordinates_max = (1.0, 1.0),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux,
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = (flux_central,
                                                                               flux_nonconservative_powell),
                                                             volume_flux_fv = surface_flux),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
