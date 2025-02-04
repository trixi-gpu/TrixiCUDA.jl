using Test
using CUDA
# Currently we need to allow scalar indexing on GPU arrays for the tests to pass,
# once the issues are resolved, this line can be removed.
CUDA.allowscalar(true)

@testset "TrixiCUDA.jl tests" begin
    @info "Starting TrixiCUDA GPU test suite"

    for (dim, path) in [("1D", "./tree_dgsem_1d/tree_dgsem_1d.jl"),
        ("2D", "./tree_dgsem_2d/tree_dgsem_2d.jl"),
        ("3D", "./tree_dgsem_3d/tree_dgsem_3d.jl")]
        @info "Running $dim DGSEM tree tests..."
        @time include(path)
        @info "Completed $dim DGSEM tree tests"
    end

    @info "All TrixiCUDA tests completed successfully"
end
