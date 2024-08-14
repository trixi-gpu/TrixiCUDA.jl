# Define common functions for solvers

# Copy data from host to device
function copy_to_device!(du::PtrArray, u::PtrArray)
    du = CuArray{Float64}(zero(du))
    u = CuArray{Float64}(u)

    return (du, u)
end

# Copy data from device to host 
function copy_to_host!(du::CuArray, u::CuArray)
    # FIXME: Maybe direct CuArray to PtrArray conversion is possible (in the future)
    du = PtrArray(Array{Float64}(du))
    u = PtrArray(Array{Float64}(u))

    return (du, u)
end
