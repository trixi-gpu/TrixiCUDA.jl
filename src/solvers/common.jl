# Define common functions for solvers

# Copy matrices from host to device
function copy_to_device!(du::PtrArray, u::PtrArray) # ? PtrArray{Float64}
    du = CuArray{Float32}(zero(du))
    u = CuArray{Float32}(u)

    return (du, u)
end

# Copy matrices from device to host 
function copy_to_host!(du::CuArray, u::CuArray) # ? CuArray{Float32}
    # ? direct CuArray to PtrArray conversion is impossible
    du = PtrArray(Array{Float64}(du))
    u = PtrArray(Array{Float64}(u))

    return (du, u)
end
