# Everything related to a DG semidiscretization in 1D.

#################################################################################################
# Functions that end with `_kernel` are CUDA kernels that are going to be launched by 
# the @cuda macro with parameters from the kernel configurator. They are purely run on 
# the device (i.e., GPU).

# Kernel for calculating fluxes along normal direction
function flux_kernel!(flux_arr, u, equations::AbstractEquations{1}, flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2) && k <= size(u, 3))
        u_node = get_node_vars(u, equations, j, k)
        flux_node = flux(u_node, 1, equations)

        for ii in axes(u, 1)
            @inbounds flux_arr[ii, j, k] = flux_node[ii]
        end
    end

    return nothing
end

# Kernel for calculating weak form
function weak_form_kernel!(du, derivative_dhat, flux_arr)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        @inbounds du[i, j, k] = zero(eltype(du)) # initialize `du` with zeros 
        for ii in axes(du, 2)
            @inbounds du[i, j, k] += derivative_dhat[j, ii] * flux_arr[i, ii, k]
        end
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating volume integrals with weak form
function flux_weak_form_kernel!(du, u, derivative_dhat,
                                equations::AbstractEquations{1}, flux::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_dhat = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_flux = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width), offset)

    # Get thread and block indices only we need to save registers
    tx, ty = threadIdx().x, threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    # Tile the computation (restrict to one tile here)
    value = zero(eltype(du))

    # Load global `derivative_dhat` into shared memory
    for ty2 in axes(du, 2)
        # Transposed access
        @inbounds shmem_dhat[ty2, ty] = derivative_dhat[ty, ty2]
    end

    # Compute flux values
    u_node = get_node_vars(u, equations, ty, k)
    flux_node = flux(u_node, 1, equations)

    @inbounds shmem_flux[tx, ty] = flux_node[tx]

    sync_threads()

    # Loop within one block to get weak form
    # TODO: Avoid potential bank conflicts
    for thread in 1:tile_width
        @inbounds value += shmem_dhat[thread, ty] * shmem_flux[tx, thread]
    end

    # Synchronization is not needed here if we use only one tile
    # sync_threads()

    # Finalize the weak form
    @inbounds du[tx, ty, k] = value

    return nothing
end

# Kernel for calculating volume fluxes
function volume_flux_kernel!(volume_flux_arr, u, equations::AbstractEquations{1},
                             volume_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^2 && k <= size(u, 3))
        j1 = div(j - 1, size(u, 2)) + 1
        j2 = rem(j - 1, size(u, 2)) + 1

        u_node = get_node_vars(u, equations, j1, k)
        u_node1 = get_node_vars(u, equations, j2, k)

        volume_flux_node = volume_flux(u_node, u_node1, 1, equations)

        for ii in axes(u, 1)
            @inbounds volume_flux_arr[ii, j1, j2, k] = volume_flux_node[ii]
        end
    end

    return nothing
end

# Kernel for calculating volume integrals
function volume_integral_kernel!(du, derivative_split, volume_flux_arr,
                                 equations::AbstractEquations{1})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        @inbounds du[i, j, k] = zero(eltype(du)) # initialize `du` with zeros
        for ii in axes(du, 2)
            @inbounds du[i, j, k] += derivative_split[j, ii] * (1 - isequal(j, ii)) * # set diagonal elements to zeros
                                     volume_flux_arr[i, j, ii, k]
        end
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating volume integrals without conservative terms
function volume_flux_integral_kernel!(du, u, derivative_split,
                                      equations::AbstractEquations{1}, volume_flux::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_split = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_value = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width), offset)

    # Get thread and block indices only we need to save registers
    ty = threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    # Tile the computation (set to one tile here)
    # Initialize the values
    for tx in axes(du, 1)
        @inbounds shmem_value[tx, ty] = zero(eltype(du))
    end

    # Load global `derivative_split` into shared memory
    for ty2 in axes(du, 2)
        # Transposed access
        @inbounds shmem_split[ty2, ty] = derivative_split[ty, ty2] *
                                         (1 - isequal(ty, ty2)) # set diagonal elements to zeros
    end

    # Synchronization is not needed here given the access pattern
    # sync_threads()

    # Compute volume fluxes
    # How to store nodes in shared memory?
    for thread in 1:tile_width
        # Volume flux is heavy in computation so we should try best to avoid redundant 
        # computation, i.e., use for loop along x direction here
        volume_flux_node = volume_flux(get_node_vars(u, equations, ty, k),
                                       get_node_vars(u, equations, thread, k),
                                       1, equations)

        # TODO: Avoid potential bank conflicts
        for tx in axes(du, 1)
            @inbounds shmem_value[tx, ty] += shmem_split[thread, ty] * volume_flux_node[tx]
        end
    end

    # Synchronization is not needed here if we use only one tile
    # sync_threads()

    # Finalize the values
    for tx in axes(du, 1)
        @inbounds du[tx, ty, k] = shmem_value[tx, ty]
    end

    return nothing
end

# Kernel for calculating symmetric and nonconservative fluxes
function noncons_volume_flux_kernel!(symmetric_flux_arr, noncons_flux_arr, u, derivative_split,
                                     equations::AbstractEquations{1}, symmetric_flux::Any,
                                     nonconservative_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^2 && k <= size(u, 3))
        j1 = div(j - 1, size(u, 2)) + 1
        j2 = rem(j - 1, size(u, 2)) + 1

        u_node = get_node_vars(u, equations, j1, k)
        u_node1 = get_node_vars(u, equations, j2, k)

        symmetric_flux_node = symmetric_flux(u_node, u_node1, 1, equations)
        noncons_flux_node = nonconservative_flux(u_node, u_node1, 1, equations)

        for ii in axes(u, 1)
            @inbounds begin
                symmetric_flux_arr[ii, j1, j2, k] = symmetric_flux_node[ii] * derivative_split[j1, j2] *
                                                    (1 - isequal(j1, j2)) # set diagonal elements to zeros                  
                noncons_flux_arr[ii, j1, j2, k] = noncons_flux_node[ii]
            end
        end
    end

    return nothing
end

# Kernel for calculating symmetric and nonconservative volume integrals
function volume_integral_kernel!(du, derivative_split, symmetric_flux_arr, noncons_flux_arr)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        @inbounds du[i, j, k] = zero(eltype(du)) # initialize `du` with zeros

        for ii in axes(du, 2)
            @inbounds du[i, j, k] += symmetric_flux_arr[i, j, ii, k] +
                                     0.5f0 *
                                     derivative_split[j, ii] * noncons_flux_arr[i, j, ii, k]
        end
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating volume integrals with conservative terms
function volume_flux_integral_kernel!(du, u, derivative_split,
                                      equations::AbstractEquations{1},
                                      symmetric_flux::Any, nonconservative_flux::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_split = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_value = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width), offset)

    # Get thread and block indices only we need to save registers
    ty = threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    # Tile the computation (set to one tile here)
    # Initialize the values
    for tx in axes(du, 1)
        @inbounds shmem_value[tx, ty] = zero(eltype(du))
    end

    # Load data from global memory into shared memory
    for ty2 in axes(du, 2)
        # Transposed access
        @inbounds shmem_split[ty2, ty] = derivative_split[ty, ty2]
    end

    # Synchronization is not needed here given the access pattern
    # sync_threads()

    # Compute volume fluxes
    # How to store nodes in shared memory?
    for thread in 1:tile_width
        # Volume flux is heavy in computation so we should try best to avoid redundant 
        # computation, i.e., use for loop along x direction here
        u_node = get_node_vars(u, equations, ty, k)
        symmetric_flux_node = symmetric_flux(u_node,
                                             get_node_vars(u, equations, thread, k),
                                             1, equations)
        noncons_flux_node = nonconservative_flux(u_node,
                                                 get_node_vars(u, equations, thread, k),
                                                 1, equations)

        # TODO: Avoid potential bank conflicts
        for tx in axes(du, 1)
            @inbounds shmem_value[tx, ty] += symmetric_flux_node[tx] * shmem_split[thread, ty] *
                                             (1 - isequal(ty, thread)) + # set diagonal elements to zeros
                                             0.5f0 *
                                             noncons_flux_node[tx] * shmem_split[thread, ty]
        end
    end

    # Synchronization is not needed here if we use only one tile
    # sync_threads()

    # Finalize the values
    for tx in axes(du, 1)
        @inbounds du[tx, ty, k] = shmem_value[tx, ty]
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume fluxes
function volume_flux_dgfv_kernel!(volume_flux_arr, fstar1_L, fstar1_R, u,
                                  alpha, atol, equations::AbstractEquations{1},
                                  volume_flux_dg::Any, volume_flux_fv::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^2 && k <= size(u, 3))
        j1 = div(j - 1, size(u, 2)) + 1
        j2 = rem(j - 1, size(u, 2)) + 1

        dg_only = isapprox(alpha[k], 0, atol = atol)

        u_node = get_node_vars(u, equations, j1, k)
        u_node1 = get_node_vars(u, equations, j2, k)

        volume_flux_node = volume_flux_dg(u_node, u_node1, 1, equations)

        for ii in axes(u, 1)
            @inbounds volume_flux_arr[ii, j1, j2, k] = volume_flux_node[ii]

            # Small optimization, no much performance gain
            if isequal(j1 + 1, j2) # avoid race condition
                flux_fv_node = volume_flux_fv(u_node, u_node1, 1, equations)

                @inbounds begin
                    fstar1_L[ii, j2, k] = flux_fv_node[ii] * (1 - dg_only)
                    fstar1_R[ii, j2, k] = flux_fv_node[ii] * (1 - dg_only)
                end
            end
        end

        # if j1 != 1 && j2 == 1 # bad
        #     u_ll = get_node_vars(u, equations, j1 - 1, k)
        #     u_rr = get_node_vars(u, equations, j1, k)
        #     flux_fv_node = volume_flux_fv(u_ll, u_rr, 1, equations)

        #     for ii in axes(u, 1)
        #         @inbounds begin
        #             fstar1_L[ii, j1, k] = flux_fv_node[ii] * (1 - dg_only)
        #             fstar1_R[ii, j1, k] = flux_fv_node[ii] * (1 - dg_only)
        #         end
        #     end
        # end
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume integrals
function volume_integral_dgfv_kernel!(du, alpha, derivative_split, inverse_weights,
                                      volume_flux_arr, fstar1_L, fstar1_R, atol,
                                      equations::AbstractEquations{1})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        @inbounds begin
            du[i, j, k] = zero(eltype(du)) # initialize `du` with zeros
            alpha_element = alpha[k]
        end

        dg_only = isapprox(alpha_element, 0, atol = atol)

        for ii in axes(du, 2)
            @inbounds du[i, j, k] += derivative_split[j, ii] *
                                     (1 - isequal(j, ii)) * # set diagonal elements to zeros
                                     volume_flux_arr[i, j, ii, k] * dg_only +
                                     (1 - alpha_element) * derivative_split[j, ii] *
                                     (1 - isequal(j, ii)) * # set diagonal elements to zeros
                                     volume_flux_arr[i, j, ii, k] * (1 - dg_only)
        end

        @inbounds du[i, j, k] += alpha_element * inverse_weights[j] *
                                 (fstar1_L[i, j + 1, k] - fstar1_R[i, j, k]) * (1 - dg_only)
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating pure DG and DG-FV volume integrals without conservative terms
function volume_flux_integral_dgfv_kernel!(du, u, alpha, atol, derivative_split, inverse_weights,
                                           equations::AbstractEquations{1},
                                           volume_flux_dg::Any, volume_flux_fv::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_split = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_fstar1 = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width + 1), offset)
    offset += sizeof(eltype(du)) * size(du, 1) * (tile_width + 1)
    shmem_value = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width), offset)

    # Get thread and block indices only we need to save registers
    ty = threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    # Load global `derivative_split` into shared memory
    for ty2 in axes(du, 2)
        # Transposed access
        @inbounds shmem_split[ty2, ty] = derivative_split[ty, ty2]
    end

    # Get variables for computation
    @inbounds alpha_element = alpha[k]
    dg_only = isapprox(alpha_element, 0, atol = atol)

    # Compute FV volume fluxes
    u_node = get_node_vars(u, equations, ty, k)
    if ty + 1 <= tile_width
        flux_fv_node = volume_flux_fv(u_node,
                                      get_node_vars(u, equations, ty + 1, k),
                                      1, equations)
    end

    # Initialize the values
    for tx in axes(du, 1)
        @inbounds begin
            # Initialize `du` with zeros
            shmem_value[tx, ty] = zero(eltype(du))
            # Initialize `fstar` side columes with zeros 
            shmem_fstar1[tx, 1] = zero(eltype(du))
            shmem_fstar1[tx, tile_width + 1] = zero(eltype(du))
        end

        if ty + 1 <= tile_width
            # Set with FV volume fluxes
            @inbounds shmem_fstar1[tx, ty + 1] = flux_fv_node[tx] * (1 - dg_only)
        end
    end

    sync_threads()

    # Contribute FV to the volume integrals
    for tx in axes(du, 1)
        @inbounds shmem_value[tx, ty] += alpha_element * inverse_weights[ty] *
                                         (shmem_fstar1[tx, ty + 1] - shmem_fstar1[tx, ty]) * (1 - dg_only)
    end

    # Compute DG volume fluxes
    for thread in 1:tile_width
        volume_flux_node = volume_flux_dg(u_node,
                                          get_node_vars(u, equations, thread, k),
                                          1, equations)

        # Contribute DG to the volume integrals
        for tx in axes(du, 1)
            @inbounds shmem_value[tx, ty] += shmem_split[thread, ty] *
                                             (1 - isequal(ty, thread)) * # set diagonal elements to zeros
                                             volume_flux_node[tx] * dg_only +
                                             (1 - alpha_element) * shmem_split[thread, ty] *
                                             (1 - isequal(ty, thread)) * # set diagonal elements to zeros
                                             volume_flux_node[tx] * (1 - dg_only)
        end
    end

    # Finalize the values
    for tx in axes(du, 1)
        @inbounds du[tx, ty, k] = shmem_value[tx, ty]
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume fluxes
function volume_flux_dgfv_kernel!(volume_flux_arr, noncons_flux_arr, fstar1_L, fstar1_R,
                                  u, alpha, atol, derivative_split,
                                  equations::AbstractEquations{1},
                                  volume_flux_dg::Any, noncons_flux_dg::Any,
                                  volume_flux_fv::Any, noncons_flux_fv::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^2 && k <= size(u, 3))
        j1 = div(j - 1, size(u, 2)) + 1
        j2 = rem(j - 1, size(u, 2)) + 1

        dg_only = isapprox(alpha[k], 0, atol = atol)

        u_node = get_node_vars(u, equations, j1, k)
        u_node1 = get_node_vars(u, equations, j2, k)

        volume_flux_node = volume_flux_dg(u_node, u_node1, 1, equations)
        noncons_flux_node = noncons_flux_dg(u_node, u_node1, 1, equations)

        for ii in axes(u, 1)
            @inbounds begin
                volume_flux_arr[ii, j1, j2, k] = volume_flux_node[ii] * derivative_split[j1, j2] *
                                                 (1 - isequal(j1, j2)) # set diagonal elements to zeros
                noncons_flux_arr[ii, j1, j2, k] = noncons_flux_node[ii]
            end

            # Small optimization, no much performance gain
            if isequal(j1 + 1, j2) # avoid race condition
                f1_node = volume_flux_fv(u_node, u_node1, 1, equations)
                f1_L_node = noncons_flux_fv(u_node, u_node1, 1, equations)
                f1_R_node = noncons_flux_fv(u_node1, u_node, 1, equations)

                @inbounds begin
                    fstar1_L[ii, j2, k] = (f1_node[ii] + 0.5f0 * f1_L_node[ii]) * (1 - dg_only)
                    fstar1_R[ii, j2, k] = (f1_node[ii] + 0.5f0 * f1_R_node[ii]) * (1 - dg_only)
                end
            end
        end

        # if j1 != 1 && j2 == 1 # bad
        #     u_ll = get_node_vars(u, equations, j1 - 1, k)
        #     u_rr = get_node_vars(u, equations, j1, k)

        #     f1_node = volume_flux_fv(u_ll, u_rr, 1, equations)

        #     f1_L_node = noncons_flux_fv(u_ll, u_rr, 1, equations)
        #     f1_R_node = noncons_flux_fv(u_rr, u_ll, 1, equations)

        #     for ii in axes(u, 1)
        #         @inbounds begin
        #             fstar1_L[ii, j1, k] = (f1_node[ii] + 0.5f0 * f1_L_node[ii]) * (1 - dg_only)
        #             fstar1_R[ii, j1, k] = (f1_node[ii] + 0.5f0 * f1_R_node[ii]) * (1 - dg_only)
        #         end
        #     end
        # end
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume integrals
function volume_integral_dgfv_kernel!(du, alpha, derivative_split, inverse_weights,
                                      volume_flux_arr, noncons_flux_arr, fstar1_L, fstar1_R,
                                      atol, equations::AbstractEquations{1})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        @inbounds begin
            du[i, j, k] = zero(eltype(du)) # initialize `du` with zeros
            alpha_element = alpha[k]
        end

        dg_only = isapprox(alpha_element, 0, atol = atol)

        for ii in axes(du, 2)
            @inbounds du[i, j, k] += (volume_flux_arr[i, j, ii, k] +
                                      0.5f0 *
                                      derivative_split[j, ii] * noncons_flux_arr[i, j, ii, k]) * dg_only +
                                     ((1 - alpha_element) * volume_flux_arr[i, j, ii, k] +
                                      0.5f0 * (1 - alpha_element) *
                                      derivative_split[j, ii] * noncons_flux_arr[i, j, ii, k]) * (1 - dg_only)
        end

        @inbounds du[i, j, k] += alpha_element * inverse_weights[j] *
                                 (fstar1_L[i, j + 1, k] - fstar1_R[i, j, k]) * (1 - dg_only)
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating pure DG and DG-FV volume integrals with conservative terms
function volume_flux_integral_dgfv_kernel!(du, u, alpha, atol, derivative_split, inverse_weights,
                                           equations::AbstractEquations{1},
                                           volume_flux_dg::Any, noncons_flux_dg::Any,
                                           volume_flux_fv::Any, noncons_flux_fv::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_split = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_fstar1 = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width + 1, 2), offset)
    offset += sizeof(eltype(du)) * size(du, 1) * (tile_width + 1)
    shmem_value = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width), offset)

    return nothing
end

# Kernel for prolonging two interfaces
function prolong_interfaces_kernel!(interfaces_u, u, neighbor_ids)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(interfaces_u, 2) && k <= size(interfaces_u, 3))
        @inbounds begin
            left_element = neighbor_ids[1, k]
            right_element = neighbor_ids[2, k]

            interfaces_u[1, j, k] = u[j, size(u, 2), left_element]
            interfaces_u[2, j, k] = u[j, 1, right_element]
        end
    end

    return nothing
end

# Kernel for calculating surface fluxes 
function surface_flux_kernel!(surface_flux_arr, interfaces_u, equations::AbstractEquations{1},
                              surface_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (j <= size(surface_flux_arr, 2))
        u_ll, u_rr = get_surface_node_vars(interfaces_u, equations, j)

        surface_flux_node = surface_flux(u_ll, u_rr, 1, equations)

        for ii in axes(surface_flux_arr, 1)
            @inbounds surface_flux_arr[ii, j] = surface_flux_node[ii]
        end
    end

    return nothing
end

# Kernel for calculating surface and both nonconservative fluxes 
function surface_noncons_flux_kernel!(surface_flux_arr, noncons_left_arr, noncons_right_arr,
                                      interfaces_u, equations::AbstractEquations{1},
                                      surface_flux::Any, nonconservative_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (j <= size(surface_flux_arr, 2))
        u_ll, u_rr = get_surface_node_vars(interfaces_u, equations, j)

        surface_flux_node = surface_flux(u_ll, u_rr, 1, equations)
        noncons_left_node = nonconservative_flux(u_ll, u_rr, 1, equations)
        noncons_right_node = nonconservative_flux(u_rr, u_ll, 1, equations)

        for ii in axes(surface_flux_arr, 1)
            @inbounds begin
                surface_flux_arr[ii, j] = surface_flux_node[ii]
                noncons_left_arr[ii, j] = noncons_left_node[ii]
                noncons_right_arr[ii, j] = noncons_right_node[ii]
            end
        end
    end

    return nothing
end

# Kernel for setting interface fluxes
function interface_flux_kernel!(surface_flux_values, surface_flux_arr, neighbor_ids)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_arr, 2))
        @inbounds begin
            left_id = neighbor_ids[1, j]
            right_id = neighbor_ids[2, j]

            surface_flux_values[i, 2, left_id] = surface_flux_arr[i, j]
            surface_flux_values[i, 1, right_id] = surface_flux_arr[i, j]
        end
    end

    return nothing
end

# Kernel for setting interface fluxes
function interface_flux_kernel!(surface_flux_values, surface_flux_arr, noncons_left_arr,
                                noncons_right_arr, neighbor_ids)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_arr, 2))
        @inbounds begin
            left_id = neighbor_ids[1, j]
            right_id = neighbor_ids[2, j]

            surface_flux_values[i, 2, left_id] = surface_flux_arr[i, j] +
                                                 0.5f0 * noncons_left_arr[i, j]
            surface_flux_values[i, 1, right_id] = surface_flux_arr[i, j] +
                                                  0.5f0 * noncons_right_arr[i, j]
        end
    end

    return nothing
end

# Kernel for prolonging two boundaries
function prolong_boundaries_kernel!(boundaries_u, u, neighbor_ids, neighbor_sides)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(boundaries_u, 2) && k <= size(boundaries_u, 3))
        @inbounds begin
            element = neighbor_ids[k]
            side = neighbor_sides[k]

            boundaries_u[1, j, k] = u[j, size(u, 2), element] * (2 - side) # set to 0 instead of NaN
            boundaries_u[2, j, k] = u[j, 1, element] * (side - 1) # set to 0 instead of NaN
        end
    end

    return nothing
end

# Kernel for calculating boundary fluxes
function boundary_flux_kernel!(surface_flux_values, boundaries_u, node_coordinates, t, boundary_arr,
                               indices_arr, neighbor_ids, neighbor_sides, orientations,
                               boundary_conditions::NamedTuple, equations::AbstractEquations{1},
                               surface_flux::Any)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (k <= length(boundary_arr))
        @inbounds begin
            boundary = boundary_arr[k]
            direction = (indices_arr[1] <= boundary) + (indices_arr[2] <= boundary)

            neighbor = neighbor_ids[boundary]
            side = neighbor_sides[boundary]
            orientation = orientations[boundary]
        end

        u_ll, u_rr = get_surface_node_vars(boundaries_u, equations, boundary)
        u_inner = (2 - side) * u_ll + (side - 1) * u_rr
        x = get_node_coords(node_coordinates, equations, boundary)

        # TODO: Improve this part
        if direction == 1
            boundary_flux_node = boundary_conditions[1](u_inner, orientation,
                                                        direction, x, t, surface_flux, equations)
        else
            boundary_flux_node = boundary_conditions[2](u_inner, orientation,
                                                        direction, x, t, surface_flux, equations)
        end

        for ii in axes(surface_flux_values, 1)
            # `boundary_flux_node` can be nothing if periodic boundary condition is applied
            @inbounds surface_flux_values[ii, direction, neighbor] = isnothing(boundary_flux_node) ? # bad
                                                                     surface_flux_values[ii,
                                                                                         direction,
                                                                                         neighbor] :
                                                                     boundary_flux_node[ii]
        end
    end

    return nothing
end

# Kernel for calculating boundary fluxes
function boundary_flux_kernel!(surface_flux_values, boundaries_u, node_coordinates, t, boundary_arr,
                               indices_arr, neighbor_ids, neighbor_sides, orientations,
                               boundary_conditions::NamedTuple, equations::AbstractEquations{1},
                               surface_flux::Any, nonconservative_flux::Any)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (k <= length(boundary_arr))
        @inbounds begin
            boundary = boundary_arr[k]
            direction = (indices_arr[1] <= boundary) + (indices_arr[2] <= boundary)

            neighbor = neighbor_ids[boundary]
            side = neighbor_sides[boundary]
            orientation = orientations[boundary]
        end

        u_ll, u_rr = get_surface_node_vars(boundaries_u, equations, boundary)
        u_inner = (2 - side) * u_ll + (side - 1) * u_rr
        x = get_node_coords(node_coordinates, equations, boundary)

        # TODO: Improve this part
        if direction == 1
            flux_node = boundary_conditions[1](u_inner, orientation, direction, x, t, surface_flux,
                                               equations)
            noncons_flux_node = boundary_conditions[1](u_inner, orientation, direction, x, t,
                                                       nonconservative_flux, equations)
        else
            flux_node = boundary_conditions[2](u_inner, orientation, direction, x, t, surface_flux,
                                               equations)
            noncons_flux_node = boundary_conditions[2](u_inner, orientation, direction, x, t,
                                                       nonconservative_flux, equations)
        end

        for ii in axes(surface_flux_values, 1)
            @inbounds surface_flux_values[ii, direction, neighbor] = flux_node[ii] +
                                                                     0.5f0 * noncons_flux_node[ii]
        end
    end

    return nothing
end

# Kernel for calculating surface integrals
function surface_integral_kernel!(du, factor_arr, surface_flux_values,
                                  equations::AbstractEquations{1})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        @inbounds begin
            du[i, j, k] -= surface_flux_values[i, 1, k] * isequal(j, 1) * factor_arr[1]
            du[i, j, k] += surface_flux_values[i, 2, k] * isequal(j, size(du, 2)) * factor_arr[2]
        end
    end

    return nothing
end

# Kernel for applying inverse Jacobian 
function jacobian_kernel!(du, inverse_jacobian, equations::AbstractEquations{1})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        @inbounds du[i, j, k] *= -inverse_jacobian[k]
    end

    return nothing
end

# Kernel for calculating source terms
function source_terms_kernel!(du, u, node_coordinates, t, equations::AbstractEquations{1},
                              source_terms::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(du, 2) && k <= size(du, 3))
        u_local = get_node_vars(u, equations, j, k)
        x_local = get_node_coords(node_coordinates, equations, j, k)

        source_terms_node = source_terms(u_local, x_local, t, equations)

        for ii in axes(du, 1)
            @inbounds du[ii, j, k] += source_terms_node[ii]
        end
    end

    return nothing
end

#################################################################################################
# Functions that begin with `cuda_` are the functions that pack CUDA kernels together to do 
# partial work in semidiscretization. They are used to invoke kernels from the host (i.e., CPU) 
# and run them on the device (i.e., GPU).

# Note that `volume_integral::VolumeIntegralPureLGLFiniteVolume` is currently experimental
# in Trixi.jl and it is not implemented here.

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{1}, nonconservative_terms,
                               equations, volume_integral::VolumeIntegralWeakForm, dg::DGSEM,
                               cache_gpu, cache_cpu)
    derivative_dhat = dg.basis.derivative_dhat

    # The maximum number of threads per block is the dominant factor when choosing the optimization 
    # method. However, there are other factors that may cause a launch failure, such as the maximum 
    # shared memory per block. Here, we have omitted all other factors, but this should be enhanced 
    # later for a safer kernel launch.

    # TODO: More checks before the kernel launch
    thread_per_block = size(du, 1) * size(du, 2)
    if thread_per_block > MAX_THREADS_PER_BLOCK
        # How to optimize when size is large (less common use)?
        flux_arr = similar(u)

        flux_kernel = @cuda launch=false flux_kernel!(flux_arr, u, equations, flux)
        flux_kernel(flux_arr, u, equations, flux;
                    kernel_configurator_2d(flux_kernel, size(u, 2), size(u, 3))...)

        weak_form_kernel = @cuda launch=false weak_form_kernel!(du, derivative_dhat, flux_arr)
        weak_form_kernel(du, derivative_dhat, flux_arr;
                         kernel_configurator_3d(weak_form_kernel, size(du)...)...)
    else
        shmem_size = (size(du, 2)^2 + size(du, 1) * size(du, 2)) * sizeof(eltype(du))
        threads = (size(du, 1), size(du, 2), 1)
        blocks = (1, 1, size(du, 3))
        @cuda threads=threads blocks=blocks shmem=shmem_size flux_weak_form_kernel!(du, u,
                                                                                    derivative_dhat,
                                                                                    equations,
                                                                                    flux)
    end

    return nothing
end

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{1}, nonconservative_terms::False,
                               equations, volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGSEM, cache_gpu, cache_cpu)
    RealT = eltype(du)

    volume_flux = volume_integral.volume_flux
    derivative_split = dg.basis.derivative_split

    thread_per_block = size(du, 2)
    if thread_per_block > MAX_THREADS_PER_BLOCK
        # How to optimize when size is large (less common use)?
        volume_flux_arr = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 3))

        volume_flux_kernel = @cuda launch=false volume_flux_kernel!(volume_flux_arr, u, equations,
                                                                    volume_flux)
        volume_flux_kernel(volume_flux_arr, u, equations, volume_flux;
                           kernel_configurator_2d(volume_flux_kernel, size(u, 2)^2, size(u, 3))...)

        volume_integral_kernel = @cuda launch=false volume_integral_kernel!(du, derivative_split,
                                                                            volume_flux_arr,
                                                                            equations)
        volume_integral_kernel(du, derivative_split, volume_flux_arr, equations;
                               kernel_configurator_3d(volume_integral_kernel, size(du)...)...)
    else
        shmem_size = (size(du, 2)^2 + size(du, 1) * size(du, 2)) * sizeof(RealT)
        threads = (1, size(du, 2), 1)
        blocks = (1, 1, size(du, 3))
        @cuda threads=threads blocks=blocks shmem=shmem_size volume_flux_integral_kernel!(du, u,
                                                                                          derivative_split,
                                                                                          equations,
                                                                                          volume_flux)
    end

    return nothing
end

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{1}, nonconservative_terms::True, equations,
                               volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM,
                               cache_gpu, cache_cpu)
    RealT = eltype(du)

    symmetric_flux, nonconservative_flux = dg.volume_integral.volume_flux
    derivative_split = dg.basis.derivative_split

    thread_per_block = size(du, 2)
    if thread_per_block > MAX_THREADS_PER_BLOCK
        # How to optimize when size is large (less common use)?
        symmetric_flux_arr = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 3))
        noncons_flux_arr = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 3))

        noncons_volume_flux_kernel = @cuda launch=false noncons_volume_flux_kernel!(symmetric_flux_arr,
                                                                                    noncons_flux_arr,
                                                                                    u,
                                                                                    derivative_split,
                                                                                    equations,
                                                                                    symmetric_flux,
                                                                                    nonconservative_flux)
        noncons_volume_flux_kernel(symmetric_flux_arr, noncons_flux_arr, u, derivative_split,
                                   equations, symmetric_flux, nonconservative_flux;
                                   kernel_configurator_2d(noncons_volume_flux_kernel,
                                                          size(u, 2)^2, size(u, 3))...)

        volume_integral_kernel = @cuda launch=false volume_integral_kernel!(du, derivative_split,
                                                                            symmetric_flux_arr,
                                                                            noncons_flux_arr)
        volume_integral_kernel(du, derivative_split, symmetric_flux_arr, noncons_flux_arr;
                               kernel_configurator_3d(volume_integral_kernel, size(du)...)...)
    else
        shmem_size = (size(du, 2)^2 + size(du, 1) * size(du, 2)) * sizeof(RealT)
        threads = (1, size(du, 2), 1)
        blocks = (1, 1, size(du, 3))
        @cuda threads=threads blocks=blocks shmem=shmem_size volume_flux_integral_kernel!(du, u,
                                                                                          derivative_split,
                                                                                          equations,
                                                                                          symmetric_flux,
                                                                                          nonconservative_flux)
    end

    return nothing
end

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{1}, nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralShockCapturingHG, dg::DGSEM,
                               cache_gpu, cache_cpu)
    RealT = eltype(du)

    volume_flux_dg, volume_flux_fv = dg.volume_integral.volume_flux_dg,
                                     dg.volume_integral.volume_flux_fv
    indicator = dg.volume_integral.indicator
    derivative_split = dg.basis.derivative_split
    inverse_weights = dg.basis.inverse_weights

    # TODO: Get copies of `u` and `du` on both device and host
    alpha = indicator(Array(u), mesh, equations, dg, cache_cpu)
    alpha = CuArray(alpha)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))

    thread_per_block = size(du, 2)
    if thread_per_block > MAX_THREADS_PER_BLOCK
        # TODO: Remove `fstar` from cache initialization
        fstar1_L = cache_gpu.fstar1_L
        fstar1_R = cache_gpu.fstar1_R

        volume_flux_arr = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 3))

        volume_flux_dgfv_kernel = @cuda launch=false volume_flux_dgfv_kernel!(volume_flux_arr, fstar1_L,
                                                                              fstar1_R, u,
                                                                              alpha, atol,
                                                                              equations, volume_flux_dg,
                                                                              volume_flux_fv)
        volume_flux_dgfv_kernel(volume_flux_arr, fstar1_L, fstar1_R, u, alpha, atol, equations,
                                volume_flux_dg, volume_flux_fv;
                                kernel_configurator_2d(volume_flux_dgfv_kernel, size(u, 2)^2,
                                                       size(u, 3))...)

        volume_integral_dgfv_kernel = @cuda launch=false volume_integral_dgfv_kernel!(du, alpha,
                                                                                      derivative_split,
                                                                                      inverse_weights,
                                                                                      volume_flux_arr,
                                                                                      fstar1_L, fstar1_R,
                                                                                      atol, equations)
        volume_integral_dgfv_kernel(du, alpha, derivative_split, inverse_weights, volume_flux_arr,
                                    fstar1_L, fstar1_R, atol, equations;
                                    kernel_configurator_3d(volume_integral_dgfv_kernel, size(du)...)...)
    else
        shmem_size = (size(du, 2)^2 + size(du, 1) * (size(du, 2) + 1) +
                      size(du, 1) * size(du, 2)) * sizeof(RealT)
        threads = (1, size(du, 2), 1)
        blocks = (1, 1, size(du, 3))
        @cuda threads=threads blocks=blocks shmem=shmem_size volume_flux_integral_dgfv_kernel!(du, u, alpha, atol,
                                                                                               derivative_split,
                                                                                               inverse_weights,
                                                                                               equations,
                                                                                               volume_flux_dg,
                                                                                               volume_flux_fv)
    end

    return nothing
end

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{1}, nonconservative_terms::True, equations,
                               volume_integral::VolumeIntegralShockCapturingHG, dg::DGSEM,
                               cache_gpu, cache_cpu)
    RealT = eltype(du)

    volume_flux_dg, noncons_flux_dg = dg.volume_integral.volume_flux_dg
    volume_flux_fv, noncons_flux_fv = dg.volume_integral.volume_flux_fv
    indicator = dg.volume_integral.indicator
    derivative_split = dg.basis.derivative_split
    inverse_weights = dg.basis.inverse_weights

    # TODO: Get copies of `u` and `du` on both device and host
    alpha = indicator(Array(u), mesh, equations, dg, cache_cpu)
    alpha = CuArray(alpha)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))

    fstar1_L = cache_gpu.fstar1_L
    fstar1_R = cache_gpu.fstar1_R

    volume_flux_arr = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 3))
    noncons_flux_arr = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 3))

    volume_flux_dgfv_kernel = @cuda launch=false volume_flux_dgfv_kernel!(volume_flux_arr,
                                                                          noncons_flux_arr,
                                                                          fstar1_L, fstar1_R, u,
                                                                          alpha, atol,
                                                                          derivative_split,
                                                                          equations,
                                                                          volume_flux_dg,
                                                                          noncons_flux_dg,
                                                                          volume_flux_fv,
                                                                          noncons_flux_fv)
    volume_flux_dgfv_kernel(volume_flux_arr, noncons_flux_arr, fstar1_L, fstar1_R, u,
                            alpha, atol, derivative_split, equations, volume_flux_dg,
                            noncons_flux_dg, volume_flux_fv, noncons_flux_fv;
                            kernel_configurator_2d(volume_flux_dgfv_kernel, size(u, 2)^2,
                                                   size(u, 3))...)

    volume_integral_dgfv_kernel = @cuda launch=false volume_integral_dgfv_kernel!(du, alpha,
                                                                                  derivative_split,
                                                                                  inverse_weights,
                                                                                  volume_flux_arr,
                                                                                  noncons_flux_arr,
                                                                                  fstar1_L, fstar1_R,
                                                                                  atol, equations)
    volume_integral_dgfv_kernel(du, alpha, derivative_split, inverse_weights, volume_flux_arr,
                                noncons_flux_arr, fstar1_L, fstar1_R, atol, equations;
                                kernel_configurator_3d(volume_integral_dgfv_kernel, size(du)...)...)

    return nothing
end

# Pack kernels for prolonging solution to interfaces
function cuda_prolong2interfaces!(u, mesh::TreeMesh{1}, equations, cache)
    neighbor_ids = cache.interfaces.neighbor_ids
    interfaces_u = cache.interfaces.u

    prolong_interfaces_kernel = @cuda launch=false prolong_interfaces_kernel!(interfaces_u, u,
                                                                              neighbor_ids)
    prolong_interfaces_kernel(interfaces_u, u, neighbor_ids;
                              kernel_configurator_2d(prolong_interfaces_kernel,
                                                     size(interfaces_u, 2),
                                                     size(interfaces_u, 3))...)

    return nothing
end

# Pack kernels for calculating interface fluxes
function cuda_interface_flux!(mesh::TreeMesh{1}, nonconservative_terms::False, equations, dg::DGSEM,
                              cache)
    RealT = eltype(cache.elements)

    surface_flux = dg.surface_integral.surface_flux

    neighbor_ids = cache.interfaces.neighbor_ids
    interfaces_u = cache.interfaces.u
    surface_flux_values = cache.elements.surface_flux_values
    surface_flux_arr = CuArray{RealT}(undef, size(interfaces_u)[2:end]...)

    surface_flux_kernel = @cuda launch=false surface_flux_kernel!(surface_flux_arr, interfaces_u,
                                                                  equations, surface_flux)
    surface_flux_kernel(surface_flux_arr, interfaces_u, equations, surface_flux;
                        kernel_configurator_1d(surface_flux_kernel, size(interfaces_u, 3))...)

    interface_flux_kernel = @cuda launch=false interface_flux_kernel!(surface_flux_values,
                                                                      surface_flux_arr,
                                                                      neighbor_ids)
    interface_flux_kernel(surface_flux_values, surface_flux_arr, neighbor_ids;
                          kernel_configurator_2d(interface_flux_kernel,
                                                 size(surface_flux_values, 1),
                                                 size(interfaces_u, 3))...)

    return nothing
end

# Pack kernels for calculating interface fluxes
function cuda_interface_flux!(mesh::TreeMesh{1}, nonconservative_terms::True, equations, dg::DGSEM,
                              cache)
    RealT = eltype(cache.elements)

    surface_flux, nonconservative_flux = dg.surface_integral.surface_flux

    neighbor_ids = cache.interfaces.neighbor_ids
    interfaces_u = cache.interfaces.u
    surface_flux_values = cache.elements.surface_flux_values

    surface_flux_arr = CuArray{RealT}(undef, size(interfaces_u)[2:end]...)
    noncons_left_arr = CuArray{RealT}(undef, size(interfaces_u)[2:end]...)
    noncons_right_arr = CuArray{RealT}(undef, size(interfaces_u)[2:end]...)

    surface_noncons_flux_kernel = @cuda launch=false surface_noncons_flux_kernel!(surface_flux_arr,
                                                                                  noncons_left_arr,
                                                                                  noncons_right_arr,
                                                                                  interfaces_u,
                                                                                  equations,
                                                                                  surface_flux,
                                                                                  nonconservative_flux)
    surface_noncons_flux_kernel(surface_flux_arr, noncons_left_arr, noncons_right_arr, interfaces_u,
                                equations, surface_flux, nonconservative_flux;
                                kernel_configurator_1d(surface_noncons_flux_kernel,
                                                       size(interfaces_u, 3))...)

    interface_flux_kernel = @cuda launch=false interface_flux_kernel!(surface_flux_values,
                                                                      surface_flux_arr,
                                                                      noncons_left_arr,
                                                                      noncons_right_arr,
                                                                      neighbor_ids)
    interface_flux_kernel(surface_flux_values, surface_flux_arr, noncons_left_arr,
                          noncons_right_arr, neighbor_ids;
                          kernel_configurator_2d(interface_flux_kernel,
                                                 size(surface_flux_values, 1),
                                                 size(interfaces_u, 3))...)

    return nothing
end

# Dummy function for asserting boundaries
function cuda_prolong2boundaries!(u, mesh::TreeMesh{1},
                                  boundary_condition::BoundaryConditionPeriodic, equations, cache)
    @assert iszero(length(cache.boundaries.orientations))
end

# Pack kernels for prolonging solution to boundaries
function cuda_prolong2boundaries!(u, mesh::TreeMesh{1}, boundary_conditions::NamedTuple, equations,
                                  cache)
    neighbor_ids = cache.boundaries.neighbor_ids
    neighbor_sides = cache.boundaries.neighbor_sides
    boundaries_u = cache.boundaries.u

    prolong_boundaries_kernel = @cuda launch=false prolong_boundaries_kernel!(boundaries_u, u,
                                                                              neighbor_ids,
                                                                              neighbor_sides)
    prolong_boundaries_kernel(boundaries_u, u, neighbor_ids, neighbor_sides;
                              kernel_configurator_2d(prolong_boundaries_kernel,
                                                     size(boundaries_u, 2),
                                                     size(boundaries_u, 3))...)

    return nothing
end

# Dummy function for asserting boundary fluxes
function cuda_boundary_flux!(t, mesh::TreeMesh{1}, boundary_condition::BoundaryConditionPeriodic,
                             nonconservative_terms, equations, dg::DGSEM, cache)
    @assert iszero(length(cache.boundaries.orientations))
end

# Pack kernels for calculating boundary fluxes
function cuda_boundary_flux!(t, mesh::TreeMesh{1}, boundary_conditions::NamedTuple,
                             nonconservative_terms::False, equations, dg::DGSEM, cache)
    surface_flux = dg.surface_integral.surface_flux

    n_boundaries_per_direction = cache.boundaries.n_boundaries_per_direction
    neighbor_ids = cache.boundaries.neighbor_ids
    neighbor_sides = cache.boundaries.neighbor_sides
    orientations = cache.boundaries.orientations
    boundaries_u = cache.boundaries.u
    node_coordinates = cache.boundaries.node_coordinates
    surface_flux_values = cache.elements.surface_flux_values

    # Create new arrays on the GPU
    lasts = zero(n_boundaries_per_direction)
    firsts = zero(n_boundaries_per_direction)

    # May introduce kernel launching overhead
    last_first_indices_kernel = @cuda launch=false last_first_indices_kernel!(lasts, firsts,
                                                                              n_boundaries_per_direction)
    last_first_indices_kernel(lasts, firsts, n_boundaries_per_direction;
                              kernel_configurator_1d(last_first_indices_kernel, length(lasts))...)

    boundary_arr = CuArray{Int}(Array(firsts)[1]:Array(lasts)[end])
    indices_arr = firsts
    boundary_conditions_callable = replace_boundary_conditions(boundary_conditions)

    boundary_flux_kernel = @cuda launch=false boundary_flux_kernel!(surface_flux_values,
                                                                    boundaries_u, node_coordinates,
                                                                    t, boundary_arr, indices_arr,
                                                                    neighbor_ids, neighbor_sides,
                                                                    orientations,
                                                                    boundary_conditions_callable,
                                                                    equations,
                                                                    surface_flux)
    boundary_flux_kernel(surface_flux_values, boundaries_u, node_coordinates, t, boundary_arr,
                         indices_arr, neighbor_ids, neighbor_sides, orientations,
                         boundary_conditions_callable, equations, surface_flux;
                         kernel_configurator_1d(boundary_flux_kernel, length(boundary_arr))...)

    return nothing
end

# Pack kernels for calculating boundary fluxes
function cuda_boundary_flux!(t, mesh::TreeMesh{1}, boundary_conditions::NamedTuple,
                             nonconservative_terms::True, equations, dg::DGSEM, cache)
    surface_flux, nonconservative_flux = dg.surface_integral.surface_flux

    n_boundaries_per_direction = cache.boundaries.n_boundaries_per_direction
    neighbor_ids = cache.boundaries.neighbor_ids
    neighbor_sides = cache.boundaries.neighbor_sides
    orientations = cache.boundaries.orientations
    boundaries_u = cache.boundaries.u
    node_coordinates = cache.boundaries.node_coordinates
    surface_flux_values = cache.elements.surface_flux_values

    # Create new arrays on the GPU
    lasts = zero(n_boundaries_per_direction)
    firsts = zero(n_boundaries_per_direction)

    # May introduce kernel launching overhead
    last_first_indices_kernel = @cuda launch=false last_first_indices_kernel!(lasts, firsts,
                                                                              n_boundaries_per_direction)
    last_first_indices_kernel(lasts, firsts, n_boundaries_per_direction;
                              kernel_configurator_1d(last_first_indices_kernel, length(lasts))...)

    boundary_arr = CuArray{Int}(Array(firsts)[1]:Array(lasts)[end])
    indices_arr = firsts
    boundary_conditions_callable = replace_boundary_conditions(boundary_conditions)

    boundary_flux_kernel = @cuda launch=false boundary_flux_kernel!(surface_flux_values,
                                                                    boundaries_u, node_coordinates,
                                                                    t, boundary_arr, indices_arr,
                                                                    neighbor_ids, neighbor_sides,
                                                                    orientations,
                                                                    boundary_conditions_callable,
                                                                    equations,
                                                                    surface_flux,
                                                                    nonconservative_flux)
    boundary_flux_kernel(surface_flux_values, boundaries_u, node_coordinates, t, boundary_arr,
                         indices_arr, neighbor_ids, neighbor_sides, orientations,
                         boundary_conditions_callable, equations, surface_flux,
                         nonconservative_flux;
                         kernel_configurator_1d(boundary_flux_kernel, length(boundary_arr))...)

    return nothing
end

# Pack kernels for calculating surface integrals
function cuda_surface_integral!(du, mesh::TreeMesh{1}, equations, dg::DGSEM, cache)
    factor_arr = CuArray([
                             dg.basis.boundary_interpolation[1, 1],
                             dg.basis.boundary_interpolation[size(du, 2), 2]
                         ])
    surface_flux_values = cache.elements.surface_flux_values

    surface_integral_kernel = @cuda launch=false surface_integral_kernel!(du, factor_arr,
                                                                          surface_flux_values,
                                                                          equations)
    surface_integral_kernel(du, factor_arr, surface_flux_values, equations;
                            kernel_configurator_3d(surface_integral_kernel, size(du)...)...)

    return nothing
end

# Pack kernels for applying Jacobian to reference element
function cuda_jacobian!(du, mesh::TreeMesh{1}, equations, cache)
    inverse_jacobian = cache.elements.inverse_jacobian

    jacobian_kernel = @cuda launch=false jacobian_kernel!(du, inverse_jacobian, equations)
    jacobian_kernel(du, inverse_jacobian, equations;
                    kernel_configurator_3d(jacobian_kernel, size(du)...)...)

    return nothing
end

# Dummy function returning nothing             
function cuda_sources!(du, u, t, source_terms::Nothing, equations::AbstractEquations{1}, cache)
    return nothing
end

# Pack kernels for calculating source terms 
function cuda_sources!(du, u, t, source_terms, equations::AbstractEquations{1}, cache)
    node_coordinates = cache.elements.node_coordinates

    source_terms_kernel = @cuda launch=false source_terms_kernel!(du, u, node_coordinates, t,
                                                                  equations, source_terms)
    source_terms_kernel(du, u, node_coordinates, t, equations, source_terms;
                        kernel_configurator_2d(source_terms_kernel, size(du, 2), size(du, 3))...)

    return nothing
end

# Put everything together into a single function.

# See also `rhs!` function in Trixi.jl
function rhs_gpu!(du, u, t, mesh::TreeMesh{1}, equations, boundary_conditions,
                  source_terms::Source, dg::DGSEM, cache_gpu, cache_cpu) where {Source}
    # reset_du!(du) 
    # reset_du!(du) is now fused into the next kernel, 
    # removing the need for a separate function call.

    cuda_volume_integral!(du, u, mesh, have_nonconservative_terms(equations), equations,
                          dg.volume_integral, dg, cache_gpu, cache_cpu)

    cuda_prolong2interfaces!(u, mesh, equations, cache_gpu)

    cuda_interface_flux!(mesh, have_nonconservative_terms(equations), equations, dg, cache_gpu)

    cuda_prolong2boundaries!(u, mesh, boundary_conditions, equations, cache_gpu)

    cuda_boundary_flux!(t, mesh, boundary_conditions,
                        have_nonconservative_terms(equations), equations, dg, cache_gpu)

    cuda_surface_integral!(du, mesh, equations, dg, cache_gpu)

    cuda_jacobian!(du, mesh, equations, cache_gpu)

    cuda_sources!(du, u, t, source_terms, equations, cache_gpu)

    return nothing
end
