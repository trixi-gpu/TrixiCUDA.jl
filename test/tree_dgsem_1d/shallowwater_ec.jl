module TestShallowWaterEc1D

using Trixi, TrixiCUDA, TrixiShallowWater
using Test

# Adapted from TrixiShallowWater.jl (66644da):
# https://github.com/trixi-framework/TrixiShallowWater.jl/blob/66644da94ebd3ad881f17379e17ffea456e15e51/examples/tree_1d_dgsem/elixir_shallowwater_ec.jl
include("../shallowwater_test_helpers.jl")

@testset "Shallow Water EC 1D" begin
    equations = ShallowWaterEquations1D(gravity = 9.81)

    function initial_condition_ec_discontinuous_bottom(x, t,
                                                       equations::ShallowWaterEquations1D)
        H = 4.25
        v = 0.0
        b = sin(x[1]) # arbitrary continuous function

        if x[1] >= 0.125 && x[1] <= 0.25
            H = 5.0
            v = 0.1882
        end

        if x[1] >= -0.25 && x[1] <= -0.125
            b = 2.0 + 0.5 * sin(2.0 * pi * x[1])
        end

        return prim2cons(SVector(H, v, b), equations)
    end

    initial_condition = initial_condition_ec_discontinuous_bottom

    volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
    solver = DGSEM(polydeg = 4,
                   surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
                   volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = -1.0
    coordinates_max = 1.0
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 4,
                    periodicity = true)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions = boundary_condition_periodic)

    test_shallowwater_rhs(semi)
end

end # module
