# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.16/examples/tree_3d_dgsem/elixir_euler_ec.jl
function (; polydeg = 3, initial_refinement_level = 3)
    return (;
            equations = CompressibleEulerEquations3D(1.4),
            initial_condition = initial_condition_weak_blast_wave,

            coordinates_min = (-2.0, -2.0, -2.0),
            coordinates_max = (2.0, 2.0, 2.0),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux = flux_ranocha,
            volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
