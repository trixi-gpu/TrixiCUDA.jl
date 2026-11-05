() -> begin
    equations = CompressibleEulerEquations2D(1.4)
    polydeg = 3
    basis = LobattoLegendreBasis(Float64, polydeg)
    indicator = IndicatorHennemannGassner(equations, basis;
                                          alpha_max = 0.5, alpha_min = 0.001,
                                          alpha_smooth = true, variable = density_pressure)
    return (;
            equations,

            coordinates_min = (-2.0, -2.0),
            coordinates_max = (2.0, 2.0),
            initial_refinement_level = 3,
            periodicity = true,

            polydeg,
            surface_flux = flux_lax_friedrichs,
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = flux_shima_etal,
                                                             volume_flux_fv = flux_lax_friedrichs),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
