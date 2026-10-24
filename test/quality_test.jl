using Aqua
using Test
using TrixiCUDA

@testset "Aqua" begin
    Aqua.test_all(TrixiCUDA)
end
