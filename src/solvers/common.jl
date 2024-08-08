# Define common functions for all solvers
# fixme: Dispatch different functions for Flaot64 and Float32 to improve performance, maybe exists a better way for Float32

# Copy matrices from host to device (Float64)
function copy_to_device!(du::PtrArray{Float64}, u::PtrArray{Float64})
    du = CUDA.zeros(size(du))
    u = CuArray{Float64}(u)

    return (du, u)
end

# Copy matrices from device to host (Float64)
function copy_to_host!(du::CuArray{Float64}, u::CuArray{Float64})
    # fixme: maybe direct PtrArray to CuArray conversion is possible
    du = PtrArray(Array(du))
    u = PtrArray(Array(u))

    return (du, u)
end
