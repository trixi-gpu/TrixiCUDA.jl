# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_euler_vortex_mortar_split.jl
function (; polydeg = 3, initial_refinement_level = 4)
    # Classical isentropic vortex of Shu (1997)
    function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
        inicenter = SVector(0.0, 0.0)
        iniamplitude = 5.0

        # Base flow
        rho = 1.0
        v1 = 1.0
        v2 = 1.0
        vel = SVector(v1, v2)
        p = 25.0
        rt = p / rho                  # ideal gas equation
        t_loc = 0.0
        cent = inicenter + vel * t_loc      # advection of center
        cent = x - cent # distance to center point
        cent = SVector(-cent[2], cent[1])
        r2 = cent[1]^2 + cent[2]^2
        du = iniamplitude / (2 * pi) * exp(0.5 * (1 - r2)) # vel. perturbation
        dtemp = -(equations.gamma - 1) / (2 * equations.gamma * rt) * du^2 # isentropic
        rho = rho * (1 + dtemp)^(1 / (equations.gamma - 1))
        vel = vel + du * cent
        v1, v2 = vel
        p = p * (1 + dtemp)^(equations.gamma / (equations.gamma - 1))
        return prim2cons(SVector(rho, v1, v2, p), equations)
    end

    return (;
            equations = CompressibleEulerEquations2D(1.4),
            initial_condition = initial_condition_isentropic_vortex,

            coordinates_min = (-10.0, -10.0),
            coordinates_max = (10.0, 10.0),
            initial_refinement_level,
            refinement_patches = ((type = "box", coordinates_min = (0.0, -10.0),
                                   coordinates_max = (10.0, 10.0)),),
            periodicity = true,

            polydeg,
            surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
            volume_integral = VolumeIntegralFluxDifferencing(flux_shima_etal),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
