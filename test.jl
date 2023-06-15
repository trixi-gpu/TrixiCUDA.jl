using CUDA, Test

function foo!(y, x)
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	stride = gridDim().x * blockDim().x

	for i ∈ index:stride:length(y)
		@inbounds y[i] += x[i]
	end

	return nothing
end

#= function configurator(kernel::CUDA.HostKernel, length::Integer)  # for 1d
	config = launch_configuration(kernel.fun)
	threads = min(length, config.threads)
	blocks = cld(length, threads)
	return (threads = threads, blocks = blocks)
end

len = 2^20
x = CUDA.fill(1.0f0, len)
y = CUDA.fill(2.0f0, len)

### inside rhs!()
kernel = @cuda name = "foo" launch = false foo!(y, x)
kernel(y, x; configurator(kernel, len)...)
@test all(Array(y) .== 3.0f0)
###
 =#

@cuda threads = 4 blocks = 2147483647 foo!(y, x)
