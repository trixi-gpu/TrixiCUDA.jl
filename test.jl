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

u0 = map(x -> Float32(0.5) * sin(pi * x) + 1, x)

function apply_surface_flux!(γ, u1, u2)
	# Get thread index 
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

	# Get loop stride
	stride = gridDim().x * blockDim().x

	# Choose to use Lax-Friedrichs flux and set equations
	surface_flux = flux_lax_friedrichs
	equations = LinearScalarAdvectionEquation1D(1.0)

	# Assign each thread to calculate corresponding flux
	for i ∈ index:stride:n_elements
		@inbounds γ[i] = surface_flux(u1[i], u2[i], 1, equations)
	end

	return nothing
end


#=  



u0 = similar(x)
println(u0)
@cuda threads = 256 initial_condition_sine_wave_gpu!(u0, x)
print(u0)
=#