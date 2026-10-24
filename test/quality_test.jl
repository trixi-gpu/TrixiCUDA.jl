using Aqua
using Test
using TrixiCUDA

@testset "Aqua" begin
    # TrixiCUDA deliberately extends Trixi for GPU arrays and adds GPU node-access helpers.
    Aqua.test_all(TrixiCUDA;
                  piracies = (treat_as_own = [TrixiCUDA.CuArray, TrixiCUDA.AbstractGPUArray,
                                  TrixiCUDA.get_node_vars, TrixiCUDA.get_node_coords,
                                  TrixiCUDA.get_surface_node_vars],))
end
