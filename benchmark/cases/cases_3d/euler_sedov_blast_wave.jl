# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_euler_sedov_blast_wave.jl
function (; polydeg = 3, initial_refinement_level = 2)
    # Sedov blast wave with self-gravity of Schlottke-Lakemper et al. (2021),
    # based on example 35.1.4 from Flash
    function initial_condition_sedov_self_gravity(x, t, equations::CompressibleEulerEquations3D)
        # Set up polar coordinates
        r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)

        r0 = 0.25 # = 4.0 * smallest dx (for domain length=8 and max-ref=7)
        E = 1.0
        p_inner = (equations.gamma - 1) * E / (4 / 3 * pi * r0^3)
        p_ambient = 1e-5 # = true Sedov setup

        # Calculate primitive variables
        # use a logistic function to transfer density value smoothly
        L = 1.0    # maximum of function
        x0 = 1.0    # center point of function
        k = -50.0 # sharpness of transfer
        logistic_function_rho = L / (1.0 + exp(-k * (r - x0)))
        rho_ambient = 1e-5
        rho = max(logistic_function_rho, rho_ambient) # clip background density to not be so tiny

        # velocities are zero
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0

        # use a logistic function to transfer pressure value smoothly
        logistic_function_p = p_inner / (1.0 + exp(-k * (r - r0)))
        p = max(logistic_function_p, p_ambient)

        return prim2cons(SVector(rho, v1, v2, v3, p), equations)
    end

    function boundary_condition_sedov_self_gravity(u_inner, orientation, direction, x, t,
                                                   surface_flux_function,
                                                   equations::CompressibleEulerEquations3D)
        # velocities are zero, density/pressure are ambient values according to
        # initial_condition_sedov_self_gravity
        rho = 1e-5
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        p = 1e-5

        u_boundary = prim2cons(SVector(rho, v1, v2, v3, p), equations)

        # Calculate boundary flux
        if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
            flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
        else # u_boundary is "left" of boundary, u_inner is "right" of boundary
            flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
        end

        return flux
    end

    equations = CompressibleEulerEquations3D(1.4)
    basis = LobattoLegendreBasis(Float64, polydeg)
    indicator = IndicatorHennemannGassner(equations, basis;
                                          alpha_max = 0.7, alpha_min = 0.001,
                                          alpha_smooth = true, variable = density_pressure)
    boundary_condition = boundary_condition_sedov_self_gravity
    return (;
            equations,
            initial_condition = initial_condition_sedov_self_gravity,

            coordinates_min = (-4.0, -4.0, -4.0),
            coordinates_max = (4.0, 4.0, 4.0),
            initial_refinement_level,
            periodicity = (false, false, false),

            polydeg,
            surface_flux = flux_hll,
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = flux_ranocha,
                                                             volume_flux_fv = flux_hll),

            boundary_conditions = (; x_neg = boundary_condition, x_pos = boundary_condition,
                                   y_neg = boundary_condition, y_pos = boundary_condition,
                                   z_neg = boundary_condition, z_pos = boundary_condition),
            source_terms = nothing,)
end
