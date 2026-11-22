# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_euler_quasi_1d_source_terms_dirichlet.jl
function (; polydeg = 4, initial_refinement_level = 4)
    boundary_condition = BoundaryConditionDirichlet(initial_condition_convergence_test)
    return (;
            equations = CompressibleEulerEquationsQuasi1D(1.4),
            initial_condition = initial_condition_convergence_test,

            coordinates_min = (-1.0,),
            coordinates_max = (1.0,),
            initial_refinement_level,
            periodicity = (false,),

            polydeg,
            surface_flux = (flux_chan_etal, flux_nonconservative_chan_etal),
            volume_integral = VolumeIntegralFluxDifferencing((flux_chan_etal,
                                                              flux_nonconservative_chan_etal)),

            boundary_conditions = (; x_neg = boundary_condition, x_pos = boundary_condition),
            source_terms = source_terms_convergence_test,)
end
