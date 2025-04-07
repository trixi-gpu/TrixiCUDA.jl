@testset "Aqua test" begin
    using Aqua

    Aqua.test_all(TrixiCUDA, piracies = false)
end
