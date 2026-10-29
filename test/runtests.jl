using TrixiCUDA
using Test
import Trixi

using CUDA
CUDA.allowscalar(false)

@testset "TrixiCUDA.jl tests" begin

    # Log testing environment
    @info "Logging testing environment information..."
    @info "Trixi preferences" log=Trixi._PREFERENCE_LOG sqrt=Trixi._PREFERENCE_SQRT loop_vectorization=Trixi._PREFERENCE_LOOPVECTORIZATION threading=Trixi._PREFERENCE_THREADING
    CUDA.versioninfo()
    println("- Multiprocessor count: ", TrixiCUDA.MULTIPROCESSOR_COUNT)
    println("- Max threads per block: ", TrixiCUDA.MAX_THREADS_PER_BLOCK)
    println("- Max shared memory per block: ", TrixiCUDA.MAX_SHARED_MEMORY_PER_BLOCK)

    @info "Starting TrixiCUDA test suite..."
    for (dim, path) in [("1D", "./tree_dgsem_1d/tree_dgsem_1d.jl"),
            ("2D", "./tree_dgsem_2d/tree_dgsem_2d.jl"),
            ("3D", "./tree_dgsem_3d/tree_dgsem_3d.jl")]
        @info "Running tests of $dim Galerkin spectral element method with tree mesh on GPU..."
        @time include(path)
        @info "Completed tests of $dim Galerkin spectral element method with tree mesh on GPU..."
    end

    # For debugging 
    # include("../test.jl")

    @info "All TrixiCUDA tests completed successfully"
end
