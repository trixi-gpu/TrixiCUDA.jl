# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_euler_shockcapturing.jl
function (; polydeg = 3, initial_refinement_level = 3)
    equations = CompressibleEulerEquations3D(1.4)
    basis = LobattoLegendreBasis(Float64, polydeg)
    indicator = IndicatorHennemannGassner(equations, basis;
                                          alpha_max = 0.5, alpha_min = 0.001,
                                          alpha_smooth = true, variable = density_pressure)
    return (;
            equations,
            initial_condition = initial_condition_weak_blast_wave,

            coordinates_min = (-2.0, -2.0, -2.0),
            coordinates_max = (2.0, 2.0, 2.0),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux = flux_ranocha,
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = flux_ranocha,
                                                             volume_flux_fv = flux_ranocha),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
