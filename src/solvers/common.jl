# Define common functions for all solvers

# Copy matrices from host to device (Float64)
function copy_to_device!(du::PtrArray{Float64}, u::PtrArray{Float64})
    du = CUDA.zeros(size(du))
    u = CuArray{Float64}(u)

    return (du, u)
end

# Copy matrices from device to host (Float64)
function copy_to_host!(du::PtrArray{Float64}, u::PtrArray{Float64})
    du = Array(du)
    u = Array(u)

    return (du, u)
end
