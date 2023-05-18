using CUDA, Test

N = 2^20
x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

function gpu_add2!(y, x)
	index = threadIdx().x    # this example only requires linear indexing, so just use `x`
	stride = blockDim().x
	for i ∈ index:stride:length(y)
		@inbounds y[i] += x[i]
	end
	return nothing
end

@cuda threads = 256 gpu_add2!(y_d, x_d)

