# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_mhd_alfven_wave_dirichlet.jl
# Fix the GLM speed explicitly; RHS benchmarks do not run GlmSpeedCallback.
function (; polydeg = 3, initial_refinement_level = 4)
    boundary_condition = BoundaryConditionDirichlet(initial_condition_convergence_test)
    return (;
            equations = IdealGlmMhdEquations2D(5.0 / 3; initial_c_h = 1.0),
            initial_condition = initial_condition_convergence_test,

            coordinates_min = (0.0, 0.0),
            coordinates_max = (sqrt(2.0), sqrt(2.0)),
            initial_refinement_level,
            periodicity = (false, false),

            polydeg,
            surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive),
                            flux_nonconservative_powell_local_symmetric),
            volume_integral = VolumeIntegralFluxDifferencing((flux_central,
                                                              flux_nonconservative_powell_local_symmetric)),

            boundary_conditions = (; x_neg = boundary_condition, x_pos = boundary_condition,
                                   y_neg = boundary_condition, y_pos = boundary_condition),
            source_terms = nothing,)
end
