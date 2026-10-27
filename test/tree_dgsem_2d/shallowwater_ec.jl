module TestShallowWaterEc2D

using Trixi, TrixiCUDA, TrixiShallowWater
using Test

# Adapted from TrixiShallowWater.jl (66644da):
# https://github.com/trixi-framework/TrixiShallowWater.jl/blob/66644da94ebd3ad881f17379e17ffea456e15e51/examples/tree_2d_dgsem/elixir_shallowwater_ec.jl
include("../shallowwater_test_helpers.jl")

@testset "Shallow Water EC 2D" begin
    equations = ShallowWaterEquations2D(gravity = 9.81)

    initial_condition = initial_condition_weak_blast_wave

    volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
    solver = DGSEM(polydeg = 4,
                   surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
                   volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-1.0, -1.0)
    coordinates_max = (1.0, 1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 2,
                    periodicity = true)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions = boundary_condition_periodic)

    function initial_condition_ec_discontinuous_bottom(x, t, element_id,
                                                       equations::ShallowWaterEquations2D)
        inicenter = SVector(0.7, 0.7)
        x_norm = x[1] - inicenter[1]
        y_norm = x[2] - inicenter[2]
        r = sqrt(x_norm^2 + y_norm^2)
        phi = atan(y_norm, x_norm)
        sin_phi, cos_phi = sincos(phi)

        H = 4.25
        v1 = 0.0
        v2 = 0.0
        b = 0.0

        if element_id == 10
            H = 5.0
            v1 = 0.1882 * cos_phi
            v2 = 0.1882 * sin_phi
        end

        if element_id == 7
            b = 2.0 + 0.5 * sin(2.0 * pi * x[1]) + 0.5 * cos(2.0 * pi * x[2])
        end

        return prim2cons(SVector(H, v1, v2, b), equations)
    end

    function initialize_ec!(u, semi)
        equations = semi.equations
        for element in eachelement(semi.solver, semi.cache)
            for j in eachnode(semi.solver), i in eachnode(semi.solver)
                x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations,
                                               semi.solver, i, j, element)
                u_node = initial_condition_ec_discontinuous_bottom(x_node, 0.0, element,
                                                                   equations)
                Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, j, element)
            end
        end
        @test any(!iszero, view(u, 4, :, :, 7))
        @test all(iszero, view(u, 4, :, :, 10))
        @test u[1, 1, 1, 10] ≈ 5.0
        return nothing
    end

    test_shallowwater_rhs(semi; initialize! = initialize_ec!)
end

end # module
