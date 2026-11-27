# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_euler_blast_wave.jl
function (; polydeg = 3, initial_refinement_level = 6)
    # Medium blast wave of Hennemann et al. (2021), Sec. 6.3
    function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations1D)
        inicenter = SVector(0.0)
        x_norm = x[1] - inicenter[1]
        r = abs(x_norm)
        cos_phi = x_norm > 0 ? 1.0 : -1.0

        rho = r > 0.5 ? 1.0 : 1.1691
        v1 = r > 0.5 ? 0.0 : 0.1882 * cos_phi
        p = r > 0.5 ? 1.0e-3 : 1.245

        return prim2cons(SVector(rho, v1, p), equations)
    end

    equations = CompressibleEulerEquations1D(1.4)
    basis = LobattoLegendreBasis(Float64, polydeg)
    indicator = IndicatorHennemannGassner(equations, basis;
                                          alpha_max = 0.5, alpha_min = 0.001,
                                          alpha_smooth = true, variable = density_pressure)
    return (;
            equations,
            initial_condition = initial_condition_blast_wave,

            coordinates_min = (-2.0,),
            coordinates_max = (2.0,),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = flux_ranocha,
                                                             volume_flux_fv = FluxLaxFriedrichs(max_abs_speed_naive)),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
