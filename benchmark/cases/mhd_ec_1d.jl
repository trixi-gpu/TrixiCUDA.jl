# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.16/examples/tree_1d_dgsem/elixir_mhd_ec.jl
function (; polydeg = 3, initial_refinement_level = 4)
    return (;
            equations = IdealGlmMhdEquations1D(2.0),
            initial_condition = initial_condition_weak_blast_wave,

            coordinates_min = (0.0,),
            coordinates_max = (1.0,),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux = flux_hindenlang_gassner,
            volume_integral = VolumeIntegralFluxDifferencing(flux_hindenlang_gassner),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
