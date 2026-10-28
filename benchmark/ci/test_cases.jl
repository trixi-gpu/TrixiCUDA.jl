# CPU-only checks of the benchmark definitions and seeded random inputs.
using Test
include("cases.jl")

@testset "GPU RHS benchmark definitions (CPU setup checks)" begin
    cases = rhs_cases()
    scripts = filter(name -> endswith(name, ".jl"), readdir(dirname(@__DIR__)))
    @test sort([c.name * ".jl" for c in cases]) == sort(scripts)
    @test length(cases) == 26
    for case in cases
        @testset "$(case.name)" begin
            semi = make_problem(case, DGSEM, SemidiscretizationHyperbolic)
            u = random_input(case, semi.equations, semi.solver, semi.cache, 123)
            @test ndims(u) == case.dim + 2
            @test all(isfinite, u)
            @test u == random_input(case, semi.equations, semi.solver, semi.cache, 123)
            @test u != random_input(case, semi.equations, semi.solver, semi.cache, 124)
            if case.variant == :mortar
                @test Trixi.nmortars(semi.solver, semi.cache) > 0
            end
            # Check every random node, including H versus h in shallow water.
            nodes = reshape(u, size(u, 1), :)
            for j in axes(nodes, 2)
                state = SVector{size(u, 1)}(nodes[:, j])
                if case.family == :euler || case.family == :mhd
                    primitive = cons2prim(state, semi.equations)
                    @test primitive[1] > 0
                    @test primitive[case.family == :euler ? case.dim + 2 : 5] > 0
                elseif case.family == :shallowwater
                    @test state[1] > 0
                end
            end
        end
    end
end
