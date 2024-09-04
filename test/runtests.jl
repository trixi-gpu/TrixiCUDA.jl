using TrixiGPU
using Test

# Note that it is complicated to get tight error bounds for GPU kernels, here we use `isapprox` 
# with the default mode to validate the precision by comparing the results from GPU kernels and 
# CPU kernels, which corresponds to requiring equality of about half of the significant digits 
# (see https://docs.julialang.org/en/v1/base/math/#Base.isapprox).

@testset "TrixiGPU.jl" begin end
