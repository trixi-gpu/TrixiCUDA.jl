import Trixi, TrixiCUDA
using Test

using CUDA
CUDA.allowscalar(false)

@testset "TrixiCUDA.jl tests" begin

    # Julia version info
    println("Julia version: ", VERSION)
    println()

    # Trixi settings and CPU execution
    @info("Trixi and CPU configuration",
          trixi_version=pkgversion(Trixi),
          cpu_threads=Threads.nthreads(),
          log=Trixi._PREFERENCE_LOG,
          sqrt=Trixi._PREFERENCE_SQRT,
          loop_vectorization=Trixi._PREFERENCE_LOOPVECTORIZATION,
          threading=Trixi._PREFERENCE_THREADING,
          timer_bars=Trixi._PREFERENCE_TIMER_BARS,)
    println()
    # Maybe TrixiShallowWater configurations are also needed here.
    # This will be implemented in the future if necessary.

    # TrixiCUDA settings and GPU hardware
    @info("TrixiCUDA and GPU information",
          multiprocessor_count=TrixiCUDA.MULTIPROCESSOR_COUNT,
          max_threads_per_block=TrixiCUDA.MAX_THREADS_PER_BLOCK,
          max_shared_memory_per_block=TrixiCUDA.MAX_SHARED_MEMORY_PER_BLOCK,)

    # CUDA version info
    println()
    CUDA.versioninfo()

    # Run tests 
    @info "Starting TrixiCUDA test suite..."
    for (dim, path) in [("1D", "./tree_dgsem_1d/tree_dgsem_1d.jl"),
            ("2D", "./tree_dgsem_2d/tree_dgsem_2d.jl"),
            ("3D", "./tree_dgsem_3d/tree_dgsem_3d.jl")]
        @info "Running tests of $dim discontinuous Galerkin spectral element method with tree mesh on GPU..."
        @time include(path)
        @info "Completed tests of $dim discontinuous Galerkin spectral element method with tree mesh on GPU..."
    end

    # For debugging 
    # include("../test.jl")

    @info "All TrixiCUDA tests completed successfully"
end
