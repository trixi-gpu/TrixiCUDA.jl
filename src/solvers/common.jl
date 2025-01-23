# Some common functions that are shared between the solvers.

# Copy data from CPU to GPU
# Function `reset_du!` is deprecated see https://github.com/trixi-gpu/TrixiCUDA.jl/pull/100
# function reset_du!(du::CuArray)
#     du_zero = zero(du)
#     du .= du_zero # no scalar indexing

#     return nothing
# end

# Kernel for getting last and first indices
# Maybe it is better to be moved to CPU
function last_first_indices_kernel!(lasts, firsts, n_boundaries_per_direction)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (i <= length(n_boundaries_per_direction))
        for ii in 1:i
            @inbounds lasts[i] += n_boundaries_per_direction[ii]
        end

        @inbounds firsts[i] = lasts[i] - n_boundaries_per_direction[i] + 1
    end

    return nothing
end
