# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_euler_taylor_green_vortex.jl
function (; polydeg = 3, initial_refinement_level = 3)
    # Classical inviscid Taylor-Green vortex
    function initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations3D)
        A = 1.0 # magnitude of speed
        Ms = 0.1 # maximum Mach number

        rho = 1.0
        v1 = A * sin(x[1]) * cos(x[2]) * cos(x[3])
        v2 = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
        v3 = 0.0
        p = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
        p = p +
            1.0 / 16.0 * A^2 * rho *
            (cos(2 * x[1]) * cos(2 * x[3]) + 2 * cos(2 * x[2]) + 2 * cos(2 * x[1]) +
             cos(2 * x[2]) * cos(2 * x[3]))

        return prim2cons(SVector(rho, v1, v2, v3, p), equations)
    end

    return (;
            equations = CompressibleEulerEquations3D(1.4),
            initial_condition = initial_condition_taylor_green_vortex,

            coordinates_min = (-1.0, -1.0, -1.0) .* pi,
            coordinates_max = (1.0, 1.0, 1.0) .* pi,
            initial_refinement_level,
            periodicity = true,

            polydeg,
            surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
            volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
