### This file is for testing aws nvidia gpu

using Trixi, LinearAlgebra, OrdinaryDiffEq, CUDA, Test

coordinates_min = -1.0
coordinates_max = 1.0
n_elements      = 16
polydeg         = 3
dx              = (coordinates_max - coordinates_min) / n_elements

basis = LobattoLegendreBasis(polydeg)

nodes = CuArray{Float32}(basis.nodes)

x_l = CuArray{Float32}(collect(0:n_elements-1)) .* Float32(dx) + CuArray{Float32}(fill(dx / 2 - 1, n_elements))

x = CuArray{Float32}(fill(1.0, polydeg + 1)) * transpose(x_l) + nodes * transpose(CuArray{Float32}(fill(1, n_elements))) .* (Float32(dx) / 2)

function initial_condition_sine_wave_gpu!(u0, x)
	index = threadIdx().x
	stride = blockDim().x
	for i ∈ index:stride:size(x)[1]
		@inbounds u0[:, i] = x[:, i]
	end
	return nothing
end

u0 = similar(x)
@cuda threads = 256 initial_condition_sine_wave_gpu!(u0, x)
println(u0[:, 15])
#= println(typeof(x)) 
u0 = similar(x)
println(u0)
@cuda threads = 256 initial_condition_sine_wave_gpu!(u0, x)
print(u0)
=#