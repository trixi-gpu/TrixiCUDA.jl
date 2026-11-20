# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_mhd_briowu_shock_tube.jl
function (; polydeg = 4, initial_refinement_level = 4)
    # Compound shock tube of Brio and Wu (1988), Sec. V
    function initial_condition_briowu_shock_tube(x, t, equations::IdealGlmMhdEquations1D)
        rho = x[1] < 0.5 ? 1.0 : 0.125
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        p = x[1] < 0.5 ? 1.0 : 0.1
        B1 = 0.75
        B2 = x[1] < 0.5 ? 1.0 : -1.0
        B3 = 0.0
        return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3), equations)
    end

    equations = IdealGlmMhdEquations1D(2.0)
    basis = LobattoLegendreBasis(Float64, polydeg)
    indicator = IndicatorHennemannGassner(equations, basis;
                                          alpha_max = 0.5, alpha_min = 0.001,
                                          alpha_smooth = true, variable = density_pressure)
    boundary_condition = BoundaryConditionDirichlet(initial_condition_briowu_shock_tube)
    return (;
            equations,
            initial_condition = initial_condition_briowu_shock_tube,

            coordinates_min = (0.0,),
            coordinates_max = (1.0,),
            initial_refinement_level,
            periodicity = (false,),

            polydeg,
            surface_flux = flux_hlle,
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = flux_derigs_etal,
                                                             volume_flux_fv = flux_hlle),

            boundary_conditions = (; x_neg = boundary_condition, x_pos = boundary_condition),
            source_terms = nothing,)
end
