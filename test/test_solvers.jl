using Trixi, TrixiGPU
using CUDA: @cuda, CuArray
using Test

# @testset "Test solver functions" begin

#     weak_form_kernel = @cuda launch=false weak_form_kernel!(du, derivative_dhat, flux_arr)
# end

advection_velocity = 1.0f0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

du = CuArray{Float64}(undef, 10, 10, 10)
derivative_dhat = CuArray{Float64}(undef, 10, 10)
flux_arr = CuArray{Float64}(undef, 10, 10, 10)

weak_form_kernel = @cuda launch=false TrixiGPU.weak_form_kernel!(du, derivative_dhat,
                                                                 flux_arr, equations)
weak_form_kernel(du,
                 derivative_dhat,
                 flux_arr, equations;
                 TrixiGPU.configurator_3d(weak_form_kernel, du)...,)
