module TestShallowWaterSourceTerms1D

using Trixi, TrixiCUDA, TrixiShallowWater
using Test

# Adapted from TrixiShallowWater.jl (66644da):
# https://github.com/trixi-framework/TrixiShallowWater.jl/blob/66644da94ebd3ad881f17379e17ffea456e15e51/examples/tree_1d_dgsem/elixir_shallowwater_source_terms.jl
include("../shallowwater_test_helpers.jl")

@testset "Shallow Water Source Terms 1D" begin
    equations = ShallowWaterEquations1D(gravity = 9.81)

    initial_condition = initial_condition_convergence_test

    surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_fjordholm_etal)
    volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
    solver = DGSEM(polydeg = 3,
                   surface_flux = surface_flux,
                   volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = 0.0
    coordinates_max = sqrt(2.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 3,
                    periodicity = true)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        source_terms = source_terms_convergence_test,
                                        boundary_conditions = boundary_condition_periodic)

    test_shallowwater_rhs(semi)
end

end # module
