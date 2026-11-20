# https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_euler_blob_mortar.jl
function (; polydeg = 3, initial_refinement_level = 4)
    # Blob test case of Agertz et al. (2007)
    function initial_condition_blob(x, t, equations::CompressibleEulerEquations2D)
        R = 1.0 # radius of the blob
        dens0 = 1.0
        Chi = 10.0 # density contrast

        tau_kh = 1.0
        tau_cr = tau_kh / 1.6 # crushing time

        velx0 = 2 * R * sqrt(Chi) / tau_cr
        vely0 = 0.0
        Ma0 = 2.7 # background flow Mach number Ma=v/c
        c = velx0 / Ma0 # sound speed

        p0 = c * c * dens0 / equations.gamma
        inicenter = SVector(-15.0, 0.0)
        x_rel = x - inicenter
        r = sqrt(x_rel[1]^2 + x_rel[2]^2)

        slope = 2.0
        dens = dens0 +
               (Chi - 1) * 0.5 * (1 + (tanh(slope * (r + R)) - (tanh(slope * (r - R)) + 1)))
        # velocity blob is zero
        velx = velx0 - velx0 * 0.5 * (1 + (tanh(slope * (r + R)) - (tanh(slope * (r - R)) + 1)))
        return prim2cons(SVector(dens, velx, vely0, p0), equations)
    end

    equations = CompressibleEulerEquations2D(5.0 / 3)
    basis = LobattoLegendreBasis(Float64, polydeg)
    indicator = IndicatorHennemannGassner(equations, basis;
                                          alpha_max = 0.05, alpha_min = 0.0001,
                                          alpha_smooth = true, variable = pressure)
    return (;
            equations,
            initial_condition = initial_condition_blob,

            coordinates_min = (-32.0, -32.0),
            coordinates_max = (32.0, 32.0),
            initial_refinement_level,
            refinement_patches = ((type = "box", coordinates_min = (-40.0, -5.0),
                                   coordinates_max = (40.0, 5.0)),
                                  (type = "box", coordinates_min = (-40.0, -5.0),
                                   coordinates_max = (40.0, 5.0))),
            periodicity = true,

            polydeg,
            surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = flux_ranocha,
                                                             volume_flux_fv = FluxLaxFriedrichs(max_abs_speed_naive)),

            boundary_conditions = boundary_condition_periodic,
            source_terms = nothing,)
end
