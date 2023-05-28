### This file is for testing aws nvidia gpu code.

using Trixi, LinearAlgebra, OrdinaryDiffEq, CUDA, Test

function gpu_add3!(y, x)
	index_i = threadIdx().x
	index_j = threadIdx().y

	stride_i = blockDim().x
	stride_j = blockDim().y

	for i ∈ index_i:stride_i:size(y, 1)
		for j ∈ index_j:stride_j:size(y, 2)
			@inbounds y[i, j] += x[i, j]
		end
	end

	return nothing
end

N = 2^10
x = CuArray(transpose(CUDA.ones(N, N)))
y = CUDA.zeros(N, N)
println(typeof(x))
@cuda threads = (16, 16) gpu_add3!(y, x)
