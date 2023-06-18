using CUDA
A = CUDA.rand(4, 4, 3)
B = CUDA.rand(4, 4)

permutedims(A, [1, 3, 2])

A = CUDA.rand(1, 1, 3)

