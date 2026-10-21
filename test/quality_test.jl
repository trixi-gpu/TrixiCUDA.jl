@testset "Aqua test" begin
    using Aqua

    include("persistent_tasks.jl")
    # Keep the persistent-task check below on the resolved dependency graph.
    Aqua.test_all(TrixiCUDA, piracies = false, persistent_tasks = false)
    @testset "Persistent tasks" begin
        @test precompile_finishes_from_manifest(TrixiCUDA)
    end
end
