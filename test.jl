using CUDA

A = rand(4, 4, 3)
B = rand(4, 4)
C = similar(A)

function matmul_slices_kernel!(C, A)
	i = threadIdx().x
	j = threadIdx().y

	@inbounds C[i, j] = A[i, j]
	return
end

A_d = CuArray(A)  # Copy A to device
B_d = CuArray(B)  # Copy B to device
C_d = CUDA.zeros(size(A))  # Initialize C on device

C_d = mapslices(x -> x * B_d, A_d, dims = [1, 2])
