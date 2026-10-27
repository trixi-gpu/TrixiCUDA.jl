module TestShallowWaterShockCapturing1D

using Trixi, TrixiCUDA, TrixiShallowWater
using Test

# Adapted from TrixiShallowWater.jl (66644da):
# https://github.com/trixi-framework/TrixiShallowWater.jl/blob/66644da94ebd3ad881f17379e17ffea456e15e51/examples/tree_1d_dgsem/elixir_shallowwater_shock_capturing.jl
include("../shallowwater_test_helpers.jl")

@testset "Shallow Water Shock Capturing 1D" begin
    equations = ShallowWaterEquations1D(gravity = 9.812, H0 = 1.75)

    function initial_condition_stone_throw_discontinuous_bottom(x, t,
                                                                equations::ShallowWaterEquations1D)

        H = equations.H0

        v = 0.0
        if x[1] >= -0.75 && x[1] <= 0.0
            v = -1.0
        elseif x[1] >= 0.0 && x[1] <= 0.75
            v = 1.0
        end

        b = (1.5 / exp(0.5 * ((x[1] - 1.0)^2)) +
             0.75 / exp(0.5 * ((x[1] + 1.0)^2)))

        if x[1] >= -1.5 && x[1] <= 0.0
            b = 0.5
        end

        return prim2cons(SVector(H, v, b), equations)
    end

    initial_condition = initial_condition_stone_throw_discontinuous_bottom

    boundary_condition = boundary_condition_slip_wall

    volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
    surface_flux = (FluxHydrostaticReconstruction(FluxLaxFriedrichs(max_abs_speed_naive),
                                                  hydrostatic_reconstruction_audusse_etal),
                    flux_nonconservative_audusse_etal)
    basis = LobattoLegendreBasis(4)

    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max = 0.5,
                                             alpha_min = 0.001,
                                             alpha_smooth = true,
                                             variable = waterheight_pressure)
    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg = volume_flux,
                                                     volume_flux_fv = surface_flux)

    solver = DGSEM(basis, surface_flux, volume_integral)

    coordinates_min = -3.0
    coordinates_max = 3.0
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 3,
                    periodicity = false)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions = boundary_condition)

    test_shallowwater_rhs(semi; boundary_conditions = boundary_condition)
end

end # module
