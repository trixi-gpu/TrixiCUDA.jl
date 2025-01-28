module TestTrixiCUDA

using Test: @testset
using CUDA
# Currently we need to allow scalar indexing on GPU arrays for the tests to pass,
# once the issues are resolved, this line can be removed.
CUDA.allowscalar(true)

@testset "TrixiCUDA.jl" begin
    # include("./tree_dgsem_1d/tree_dgsem_1d.jl")
    # include("./tree_dgsem_2d/tree_dgsem_2d.jl")
    # include("./tree_dgsem_3d/tree_dgsem_3d.jl")
end

end # module
