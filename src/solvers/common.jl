# Define common functions for solvers

# Copy matrices from host to device
function copy_to_device!(du::PtrArray, u::PtrArray) # ? PtrArray{Float64}
    du = CuArray{Float32}(zero(du))
    u = CuArray{Float32}(u)

    return (du, u)
end

# Copy matrices from device to host 
function copy_to_host!(du::CuArray, u::CuArray) # ? CuArray{Float32}
    # ? direct CuArray to PtrArray conversion is possible
    du = PtrArray{Float64}(Array(du))
    u = PtrArray{Float64}(Array(u))

    return (du, u)
end
