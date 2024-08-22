# Some common functions that are shared between the solvers.

# Replace the `boundary_condition_periodic` from Trixi.jl with a callable one
function replace_boundary_conditions(boundary_conditions::NamedTuple)
    keys_ = keys(boundary_conditions)
    values_ = (func == boundary_condition_periodic ? boundary_condition_periodic_callable : func
               for func in values(boundary_conditions))
    return NamedTuple{keys_}(values_)
end

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

# Kernel for getting last and first indices
function last_first_indices_kernel!(lasts, firsts, n_boundaries_per_direction)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (i <= length(n_boundaries_per_direction))
        @inbounds begin
            for ii in 1:i
                lasts[i] += n_boundaries_per_direction[ii]
            end
            firsts[i] = lasts[i] - n_boundaries_per_direction[i] + 1
        end
    end

    return nothing
end
