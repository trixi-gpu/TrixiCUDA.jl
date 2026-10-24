using Aqua
using Test
using TrixiCUDA

include("persistent_tasks.jl")

@testset "Aqua" begin
    # TrixiCUDA deliberately extends Trixi for GPU arrays and adds GPU node-access helpers.
    Aqua.test_all(TrixiCUDA;
                  # Check task shutdown below using the resolved dependency graph.
                  persistent_tasks = false,
                  piracies = (treat_as_own = [TrixiCUDA.CuArray, TrixiCUDA.AbstractGPUArray,
                                  TrixiCUDA.get_node_vars, TrixiCUDA.get_node_coords,
                                  TrixiCUDA.get_surface_node_vars],))
    @testset "Persistent tasks" begin
        @test precompile_finishes_from_manifest(TrixiCUDA)
    end
end
