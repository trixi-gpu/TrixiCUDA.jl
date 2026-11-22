# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_euler_quasi_1d_discontinuous.jl
function (; polydeg = 3, initial_refinement_level = 6)
    # Discontinuous state of Chan et al. (2024)
    function initial_condition_discontinuity(x, t, equations::CompressibleEulerEquationsQuasi1D)
        rho = x[1] < 0 ? 3.4718 : 2.0
        v1 = x[1] < 0 ? -2.5923 : -3.0
        p = x[1] < 0 ? 5.7118 : 2.639
        a = x[1] < 0 ? 1.0 : 1.5
        return prim2cons(SVector(rho, v1, p, a), equations)
    end

    equations = CompressibleEulerEquationsQuasi1D(1.4)
    basis = LobattoLegendreBasis(Float64, polydeg)
    indicator = IndicatorHennemannGassner(equations, basis;
                                          alpha_max = 0.5, alpha_min = 0.001,
                                          alpha_smooth = true, variable = density_pressure)
    surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_chan_etal)
    return (;
            equations,
            initial_condition = initial_condition_discontinuity,

            coordinates_min = (-1.0,),
            coordinates_max = (1.0,),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux,
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = (flux_chan_etal,
                                                                               flux_nonconservative_chan_etal),
                                                             volume_flux_fv = surface_flux),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
