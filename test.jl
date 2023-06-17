using CUDA

a = CUDA.rand(4, 4)
b = CUDA.rand(4, 4)
c = CUDA.zeros(4)

function foo!(a, b, c)
	i = threadIdx().x

	@inbounds c[i] = transpose(a[i, :]) * b[:, i]
	return nothing
end

#= @cuda threads = 4 foo!(a, b, c) =#

CUDA.dot(a[2, :], b[:, 2])