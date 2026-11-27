# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_mhd_ec_shockcapturing.jl
# Fix the GLM speed explicitly; RHS benchmarks do not run GlmSpeedCallback.
function (; polydeg = 4, initial_refinement_level = 3)
    equations = IdealGlmMhdEquations3D(1.4; initial_c_h = 1.0)
    basis = LobattoLegendreBasis(Float64, polydeg)
    indicator = IndicatorHennemannGassner(equations, basis;
                                          alpha_max = 0.5, alpha_min = 0.001,
                                          alpha_smooth = true, variable = density_pressure)
    surface_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
    volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
    return (;
            equations,
            initial_condition = initial_condition_weak_blast_wave,

            coordinates_min = (-2.0, -2.0, -2.0),
            coordinates_max = (2.0, 2.0, 2.0),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux,
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = volume_flux,
                                                             volume_flux_fv = surface_flux),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
