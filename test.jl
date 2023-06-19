using CUDA

function foo!(arr, arr2, arr3, equations::AbstractEquations, surface_flux::Function)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(arr, 2) && k <= size(arr, 3))
        @inbounds arr[1, j, k] = surface_flux(arr2[1, j, k], arr3[1, j, k], 1, equations)
    end
    return nothing
end

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

arr2 = CUDA.rand(1, 4, 4)
arr3 = CUDA.rand(1, 4, 4)
arr = similar(arr2)

@cuda threads = (2, 2) blocks = (2, 2) foo!(arr, arr2, arr3, equations, surface_flux)