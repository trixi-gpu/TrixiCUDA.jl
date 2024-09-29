include("test_macros.jl")

a = [1, 2]
b = [1, 2]

@test_approx (a, b)
