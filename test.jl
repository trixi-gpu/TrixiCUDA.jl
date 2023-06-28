using CUDA, Test, BenchmarkTools, StaticArrays

# Cause errors since access arrays within matrix
######################################################################
function foo1!(A, B)
    i = threadIdx().x

    @inbounds begin
        a = zeros(4)
        for ii in 1:4
            a[ii] = B[ii, i]
        end
        A[i] = sum(a)
    end

    return nothing
end

A = CUDA.rand(4)
B = CUDA.rand(4, 4)
@cuda threads = 4 foo1!(A, B)

# Works since pack arrays into single element in matrix
######################################################################
function foo2!(A, B)
    i = threadIdx().x
    j = threadIdx().y

    @inbounds begin
        A[i, j] = sum(B[i, j])
    end

    return nothing
end

A = CUDA.zeros(4, 4)

matrix = Array{SVector{4,Float32}}(undef, 4, 4)
for i in 1:4
    for j in 1:4
        matrix[i, j] = SVector(1.0, 2.0, 3.0, 4.0)
    end
end

B = CuArray(matrix)

@cuda threads = (4, 4) foo2!(A, B)