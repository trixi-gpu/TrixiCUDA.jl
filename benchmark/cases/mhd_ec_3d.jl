# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.16/examples/tree_3d_dgsem/elixir_mhd_ec.jl
# Fix the GLM speed explicitly; RHS benchmarks do not run GlmSpeedCallback.
function (; polydeg = 3, initial_refinement_level = 2)
    return (;
            equations = IdealGlmMhdEquations3D(1.4; initial_c_h = 1.0),
            initial_condition = initial_condition_weak_blast_wave,

            coordinates_min = (-2.0, -2.0, -2.0),
            coordinates_max = (2.0, 2.0, 2.0),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux = (flux_hindenlang_gassner, flux_nonconservative_powell),
            volume_integral = VolumeIntegralFluxDifferencing((flux_hindenlang_gassner,
                                                              flux_nonconservative_powell)),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
