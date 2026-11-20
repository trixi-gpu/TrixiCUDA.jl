# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_euler_sedov_blast_wave.jl
function (; polydeg = 3, initial_refinement_level = 6)
    # Sedov blast wave based on example 35.1.4 from Flash
    function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)
        # Set up polar coordinates
        inicenter = SVector(0.0, 0.0)
        x_norm = x[1] - inicenter[1]
        y_norm = x[2] - inicenter[2]
        r = sqrt(x_norm^2 + y_norm^2)

        r0 = 0.21875 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
        E = 1.0
        p0_inner = 3 * (equations.gamma - 1) * E / (3 * pi * r0^2)
        p0_outer = 1.0e-5 # = true Sedov setup

        # Calculate primitive variables
        rho = 1.0
        v1 = 0.0
        v2 = 0.0
        p = r > r0 ? p0_outer : p0_inner

        return prim2cons(SVector(rho, v1, v2, p), equations)
    end

    equations = CompressibleEulerEquations2D(1.4)
    basis = LobattoLegendreBasis(Float64, polydeg)
    indicator = IndicatorHennemannGassner(equations, basis;
                                          alpha_max = 0.5, alpha_min = 0.001,
                                          alpha_smooth = true, variable = density_pressure)
    return (;
            equations,
            initial_condition = initial_condition_sedov_blast_wave,

            coordinates_min = (-2.0, -2.0),
            coordinates_max = (2.0, 2.0),
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = flux_chandrashekar,
                                                             volume_flux_fv = FluxLaxFriedrichs(max_abs_speed_naive)),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
