# GPU kernels related to a DG semidiscretization in 2D.

# Functions that end with `_kernel` are CUDA kernels that are going to be launched by 
# the @cuda macro with parameters from the kernel configurator. They are purely run on 
# the device (i.e., GPU).

# Kernel for calculating fluxes along normal directions
function flux_kernel!(flux_arr1, flux_arr2, u, equations::AbstractEquations{2}, flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^2 && k <= size(u, 4))
        j1 = div(j - 1, size(u, 2)) + 1
        j2 = rem(j - 1, size(u, 2)) + 1

        u_node = get_node_vars(u, equations, j1, j2, k)

        flux_node1 = flux(u_node, 1, equations)
        flux_node2 = flux(u_node, 2, equations)

        for ii in axes(u, 1)
            @inbounds begin
                flux_arr1[ii, j1, j2, k] = flux_node1[ii]
                flux_arr2[ii, j1, j2, k] = flux_node2[ii]
            end
        end
    end

    return nothing
end

# Kernel for calculating weak form
function weak_form_kernel!(du, derivative_dhat, flux_arr1, flux_arr2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 4))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds du[i, j1, j2, k] = zero(eltype(du)) # initialize `du` with zeros

        for ii in axes(du, 2)
            @inbounds du[i, j1, j2, k] += derivative_dhat[j1, ii] * flux_arr1[i, ii, j2, k] +
                                          derivative_dhat[j2, ii] * flux_arr2[i, j1, ii, k]
        end
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating volume integrals with weak form
function flux_weak_form_kernel!(du, u, derivative_dhat,
                                equations::AbstractEquations{2}, flux::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_dhat = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_flux = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width, tile_width, 2), offset)

    # Get thread and block indices only we need to save registers
    tx, ty = threadIdx().x, threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ty1 = div(ty - 1, tile_width) + 1
    ty2 = rem(ty - 1, tile_width) + 1

    # Tile the computation (restrict to one tile here)
    value = zero(eltype(du))

    # Load global `derivative_dhat` into shared memory
    # Transposed load
    @inbounds shmem_dhat[ty1, ty2] = derivative_dhat[ty2, ty1]

    # Compute flux values
    u_node = get_node_vars(u, equations, ty1, ty2, k)
    flux_node1 = flux(u_node, 1, equations)
    flux_node2 = flux(u_node, 2, equations)

    @inbounds begin
        shmem_flux[tx, ty1, ty2, 1] = flux_node1[tx]
        shmem_flux[tx, ty1, ty2, 2] = flux_node2[tx]
    end

    sync_threads()

    # Loop within one block to get weak form
    # TODO: Avoid potential bank conflicts
    for thread in 1:tile_width
        @inbounds value += shmem_dhat[thread, ty1] * shmem_flux[tx, thread, ty2, 1] +
                           shmem_dhat[thread, ty2] * shmem_flux[tx, ty1, thread, 2]
    end

    # Synchronization is not needed here if we use only one tile
    # sync_threads()

    # Finalize the weak form
    @inbounds du[tx, ty1, ty2, k] = value

    return nothing
end

# Kernel for calculating volume fluxes
function volume_flux_kernel!(volume_flux_arr1, volume_flux_arr2, u, equations::AbstractEquations{2},
                             volume_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^3 && k <= size(u, 4))
        u2 = size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        u_node = get_node_vars(u, equations, j1, j2, k)
        u_node1 = get_node_vars(u, equations, j3, j2, k)
        u_node2 = get_node_vars(u, equations, j1, j3, k)

        volume_flux_node1 = volume_flux(u_node, u_node1, 1, equations)
        volume_flux_node2 = volume_flux(u_node, u_node2, 2, equations)

        for ii in axes(u, 1)
            @inbounds begin
                volume_flux_arr1[ii, j1, j3, j2, k] = volume_flux_node1[ii]
                volume_flux_arr2[ii, j1, j2, j3, k] = volume_flux_node2[ii]
            end
        end
    end

    return nothing
end

# Kernel for calculating volume integrals
function volume_integral_kernel!(du, derivative_split, volume_flux_arr1, volume_flux_arr2,
                                 equations::AbstractEquations{2})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 4))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds du[i, j1, j2, k] = zero(eltype(du)) # initialize `du` with zeros

        for ii in axes(du, 2)
            @inbounds du[i, j1, j2, k] += volume_flux_arr1[i, j1, ii, j2, k] * derivative_split[j1, ii] *
                                          (1 - isequal(j1, ii)) + # set diagonal elements to zeros
                                          volume_flux_arr2[i, j1, j2, ii, k] * derivative_split[j2, ii] *
                                          (1 - isequal(j2, ii)) # set diagonal elements to zeros
        end
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating volume integrals without conservative terms
function volume_flux_integral_kernel!(du, u, derivative_split,
                                      equations::AbstractEquations{2}, volume_flux::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_split = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_value = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width, tile_width), offset)

    # Get thread and block indices only we need save registers
    ty = threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ty1 = div(ty - 1, tile_width) + 1
    ty2 = rem(ty - 1, tile_width) + 1

    # Tile the computation (set to one tile here)
    # Initialize the values
    for tx in axes(du, 1)
        @inbounds shmem_value[tx, ty1, ty2] = zero(eltype(du))
    end

    # Load global `derivative_split` into shared memory
    # Transposed load
    @inbounds shmem_split[ty1, ty2] = derivative_split[ty2, ty1] *
                                      (1 - isequal(ty1, ty2)) # set diagonal elements to zeros

    sync_threads()

    # Compute volume fluxes
    # How to store nodes in shared memory?
    for thread in 1:tile_width
        # Volume flux is heavy in computation so we should try best to avoid redundant 
        # computation, i.e., use for loop along x direction here
        u_node = get_node_vars(u, equations, ty1, ty2, k)
        volume_flux_node1 = volume_flux(u_node,
                                        get_node_vars(u, equations, thread, ty2, k),
                                        1, equations)
        volume_flux_node2 = volume_flux(u_node,
                                        get_node_vars(u, equations, ty1, thread, k),
                                        2, equations)

        # TODO: Avoid potential bank conflicts 
        # Try another way to parallelize (ty1, ty2) with threads to ty3, then 
        # consolidate each computation back to (ty1, ty2)
        for tx in axes(du, 1)
            @inbounds shmem_value[tx, ty1, ty2] += shmem_split[thread, ty1] * volume_flux_node1[tx] +
                                                   shmem_split[thread, ty2] * volume_flux_node2[tx]
        end
    end

    # Synchronization is not needed here if we use only one tile
    # sync_threads()

    # Finalize the values
    for tx in axes(du, 1)
        @inbounds du[tx, ty1, ty2, k] = shmem_value[tx, ty1, ty2]
    end

    return nothing
end

# Kernel for calculating symmetric and nonconservative fluxes
function noncons_volume_flux_kernel!(symmetric_flux_arr1, symmetric_flux_arr2, noncons_flux_arr1,
                                     noncons_flux_arr2, u, derivative_split,
                                     equations::AbstractEquations{2}, symmetric_flux::Any,
                                     nonconservative_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^3 && k <= size(u, 4))
        u2 = size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        u_node = get_node_vars(u, equations, j1, j2, k)
        u_node1 = get_node_vars(u, equations, j3, j2, k)
        u_node2 = get_node_vars(u, equations, j1, j3, k)

        symmetric_flux_node1 = symmetric_flux(u_node, u_node1, 1, equations)
        symmetric_flux_node2 = symmetric_flux(u_node, u_node2, 2, equations)

        noncons_flux_node1 = nonconservative_flux(u_node, u_node1, 1, equations)
        noncons_flux_node2 = nonconservative_flux(u_node, u_node2, 2, equations)

        for ii in axes(u, 1)
            @inbounds begin
                symmetric_flux_arr1[ii, j1, j3, j2, k] = symmetric_flux_node1[ii] * derivative_split[j1, j3] *
                                                         (1 - isequal(j1, j3)) # set diagonal elements to zeros
                symmetric_flux_arr2[ii, j1, j2, j3, k] = symmetric_flux_node2[ii] * derivative_split[j2, j3] *
                                                         (1 - isequal(j2, j3)) # set diagonal elements to zeros

                noncons_flux_arr1[ii, j1, j3, j2, k] = noncons_flux_node1[ii]
                noncons_flux_arr2[ii, j1, j2, j3, k] = noncons_flux_node2[ii]
            end
        end
    end

    return nothing
end

# Kernel for calculating symmetric and nonconservative volume integrals
function volume_integral_kernel!(du, derivative_split, symmetric_flux_arr1, symmetric_flux_arr2,
                                 noncons_flux_arr1, noncons_flux_arr2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 4))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds du[i, j1, j2, k] = zero(eltype(du)) # initialize `du` with zeros

        for ii in axes(du, 2)
            @inbounds du[i, j1, j2, k] += symmetric_flux_arr1[i, j1, ii, j2, k] +
                                          symmetric_flux_arr2[i, j1, j2, ii, k] +
                                          0.5f0 *
                                          derivative_split[j1, ii] * noncons_flux_arr1[i, j1, ii, j2, k] +
                                          0.5f0 *
                                          derivative_split[j2, ii] * noncons_flux_arr2[i, j1, j2, ii, k]
        end
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating volume integrals with conservative terms
function volume_flux_integral_kernel!(du, u, derivative_split,
                                      equations::AbstractEquations{2},
                                      symmetric_flux::Any, nonconservative_flux::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_split = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_value = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width, tile_width), offset)

    # Get thread and block indices only we need save registers
    ty = threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ty1 = div(ty - 1, tile_width) + 1
    ty2 = rem(ty - 1, tile_width) + 1

    # Tile the computation (set to one tile here)
    # Initialize the values
    for tx in axes(du, 1)
        @inbounds shmem_value[tx, ty1, ty2] = zero(eltype(du))
    end

    # Load data from global memory into shared memory
    # Transposed load
    @inbounds shmem_split[ty1, ty2] = derivative_split[ty2, ty1]

    sync_threads()

    # Compute volume fluxes
    # How to store nodes in shared memory?
    for thread in 1:tile_width
        # Volume flux is heavy in computation so we should try best to avoid redundant 
        # computation, i.e., use for loop along x direction here
        u_node = get_node_vars(u, equations, ty1, ty2, k)
        symmetric_flux_node1 = symmetric_flux(u_node,
                                              get_node_vars(u, equations, thread, ty2, k),
                                              1, equations)
        symmetric_flux_node2 = symmetric_flux(u_node,
                                              get_node_vars(u, equations, ty1, thread, k),
                                              2, equations)
        noncons_flux_node1 = nonconservative_flux(u_node,
                                                  get_node_vars(u, equations, thread, ty2, k),
                                                  1, equations)
        noncons_flux_node2 = nonconservative_flux(u_node,
                                                  get_node_vars(u, equations, ty1, thread, k),
                                                  2, equations)

        # TODO: Avoid potential bank conflicts
        for tx in axes(du, 1)
            @inbounds shmem_value[tx, ty1, ty2] += symmetric_flux_node1[tx] * shmem_split[thread, ty1] *
                                                   (1 - isequal(ty1, thread)) + # set diagonal elements to zeros
                                                   symmetric_flux_node2[tx] * shmem_split[thread, ty2] *
                                                   (1 - isequal(ty2, thread)) + # set diagonal elements to zeros
                                                   0.5f0 *
                                                   noncons_flux_node1[tx] * shmem_split[thread, ty1] +
                                                   0.5f0 *
                                                   noncons_flux_node2[tx] * shmem_split[thread, ty2]
        end
    end

    # Synchronization is not needed here if we use only one tile
    # sync_threads()

    # Finalize the values
    for tx in axes(du, 1)
        @inbounds du[tx, ty1, ty2, k] = shmem_value[tx, ty1, ty2]
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume fluxes
function volume_flux_dgfv_kernel!(volume_flux_arr1, volume_flux_arr2, fstar1_L, fstar1_R,
                                  fstar2_L, fstar2_R, u, alpha, atol,
                                  equations::AbstractEquations{2},
                                  volume_flux_dg::Any, volume_flux_fv::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^3 && k <= size(u, 4))
        u2 = size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        dg_only = isapprox(alpha[k], 0, atol = atol)

        u_node = get_node_vars(u, equations, j1, j2, k)
        u_node1 = get_node_vars(u, equations, j3, j2, k)
        u_node2 = get_node_vars(u, equations, j1, j3, k)

        volume_flux_node1 = volume_flux_dg(u_node, u_node1, 1, equations)
        volume_flux_node2 = volume_flux_dg(u_node, u_node2, 2, equations)

        for ii in axes(u, 1)
            @inbounds begin
                volume_flux_arr1[ii, j1, j3, j2, k] = volume_flux_node1[ii]
                volume_flux_arr2[ii, j1, j2, j3, k] = volume_flux_node2[ii]
            end

            # Small optimization, no much performance gain
            if isequal(j1 + 1, j3) # avoid race condition
                flux_fv_node1 = volume_flux_fv(u_node, u_node1, 1, equations)

                @inbounds begin
                    fstar1_L[ii, j3, j2, k] = flux_fv_node1[ii] * (1 - dg_only)
                    fstar1_R[ii, j3, j2, k] = flux_fv_node1[ii] * (1 - dg_only)
                end
            end

            if isequal(j2 + 1, j3) # avoid race condition
                flux_fv_node2 = volume_flux_fv(u_node, u_node2, 2, equations)

                @inbounds begin
                    fstar2_L[ii, j1, j3, k] = flux_fv_node2[ii] * (1 - dg_only)
                    fstar2_R[ii, j1, j3, k] = flux_fv_node2[ii] * (1 - dg_only)
                end
            end
        end

        # if j1 != 1 && j3 == 1 # bad
        #     u_ll = get_node_vars(u, equations, j1 - 1, j2, k)
        #     u_rr = get_node_vars(u, equations, j1, j2, k)
        #     flux_fv_node1 = volume_flux_fv(u_ll, u_rr, 1, equations)

        #     for ii in axes(u, 1)
        #         @inbounds begin
        #             fstar1_L[ii, j1, j2, k] = flux_fv_node1[ii] * (1 - dg_only)
        #             fstar1_R[ii, j1, j2, k] = flux_fv_node1[ii] * (1 - dg_only)
        #         end
        #     end
        # end

        # if j2 != 1 && j3 == 1 # bad
        #     u_ll = get_node_vars(u, equations, j1, j2 - 1, k)
        #     u_rr = get_node_vars(u, equations, j1, j2, k)
        #     flux_fv_node2 = volume_flux_fv(u_ll, u_rr, 2, equations)

        #     for ii in axes(u, 1)
        #         @inbounds begin
        #             fstar2_L[ii, j1, j2, k] = flux_fv_node2[ii] * (1 - dg_only)
        #             fstar2_R[ii, j1, j2, k] = flux_fv_node2[ii] * (1 - dg_only)
        #         end
        #     end
        # end
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume integrals
function volume_integral_dgfv_kernel!(du, alpha, derivative_split, inverse_weights,
                                      volume_flux_arr1, volume_flux_arr2,
                                      fstar1_L, fstar1_R, fstar2_L, fstar2_R, atol,
                                      equations::AbstractEquations{2})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 4))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds begin
            du[i, j1, j2, k] = zero(eltype(du)) # initialize `du` with zeros
            alpha_element = alpha[k]
        end

        dg_only = isapprox(alpha_element, 0, atol = atol)

        for ii in axes(du, 2)
            @inbounds du[i, j1, j2, k] += (derivative_split[j1, ii] *
                                           (1 - isequal(j1, ii)) * # set diagonal elements to zeros
                                           volume_flux_arr1[i, j1, ii, j2, k] +
                                           derivative_split[j2, ii] *
                                           (1 - isequal(j2, ii)) * # set diagonal elements to zeros
                                           volume_flux_arr2[i, j1, j2, ii, k]) * dg_only +
                                          ((1 - alpha_element) * derivative_split[j1, ii] *
                                           (1 - isequal(j1, ii)) * # set diagonal elements to zeros
                                           volume_flux_arr1[i, j1, ii, j2, k] +
                                           (1 - alpha_element) * derivative_split[j2, ii] *
                                           (1 - isequal(j2, ii)) * # set diagonal elements to zeros
                                           volume_flux_arr2[i, j1, j2, ii, k]) * (1 - dg_only)
        end

        @inbounds du[i, j1, j2, k] += alpha_element *
                                      (inverse_weights[j1] *
                                       (fstar1_L[i, j1 + 1, j2, k] - fstar1_R[i, j1, j2, k]) +
                                       inverse_weights[j2] *
                                       (fstar2_L[i, j1, j2 + 1, k] - fstar2_R[i, j1, j2, k])) * (1 - dg_only)
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating pure DG and DG-FV volume integrals without conservative terms
function volume_flux_integral_dgfv_kernel!(du, u, alpha, atol, derivative_split, inverse_weights,
                                           equations::AbstractEquations{2},
                                           volume_flux_dg::Any, volume_flux_fv::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    # TODO: Combine `fstar` into single allocation
    shmem_split = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_fstar1 = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width + 1, tile_width), offset)
    offset += sizeof(eltype(du)) * size(du, 1) * (tile_width + 1) * tile_width
    shmem_fstar2 = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width, tile_width + 1), offset)
    offset += sizeof(eltype(du)) * size(du, 1) * tile_width * (tile_width + 1)
    shmem_value = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width, tile_width), offset)

    # Get thread and block indices only we need save registers
    ty = threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ty1 = div(ty - 1, tile_width) + 1
    ty2 = rem(ty - 1, tile_width) + 1

    # Load global `derivative_split` into shared memory
    # Transposed load
    @inbounds shmem_split[ty1, ty2] = derivative_split[ty2, ty1]

    # Get variables for computation
    @inbounds alpha_element = alpha[k]
    dg_only = isapprox(alpha_element, 0, atol = atol)

    # Compute FV volume fluxes
    u_node = get_node_vars(u, equations, ty1, ty2, k)
    if ty1 + 1 <= tile_width
        flux_fv_node1 = volume_flux_fv(u_node,
                                       get_node_vars(u, equations, ty1 + 1, ty2, k),
                                       1, equations)
    end
    if ty2 + 1 <= tile_width
        flux_fv_node2 = volume_flux_fv(u_node,
                                       get_node_vars(u, equations, ty1, ty2 + 1, k),
                                       2, equations)
    end

    # Initialize the values
    for tx in axes(du, 1)
        @inbounds begin
            # Initialize `du` with zeros
            shmem_value[tx, ty1, ty2] = zero(eltype(du))
            # Initialize `fstar` side columes with zeros 
            shmem_fstar1[tx, 1, ty2] = zero(eltype(du))
            shmem_fstar1[tx, tile_width + 1, ty2] = zero(eltype(du))
            shmem_fstar2[tx, ty1, 1] = zero(eltype(du))
            shmem_fstar2[tx, ty1, tile_width + 1] = zero(eltype(du))
        end

        if ty1 + 1 <= tile_width
            # Set with FV volume fluxes
            @inbounds shmem_fstar1[tx, ty1 + 1, ty2] = flux_fv_node1[tx] * (1 - dg_only)
        end
        if ty2 + 1 <= tile_width
            # Set with FV volume fluxes
            @inbounds shmem_fstar2[tx, ty1, ty2 + 1] = flux_fv_node2[tx] * (1 - dg_only)
        end
    end

    sync_threads()

    # Contribute FV to the volume integrals
    for tx in axes(du, 1)
        @inbounds shmem_value[tx, ty1, ty2] += alpha_element *
                                               (inverse_weights[ty1] *
                                                (shmem_fstar1[tx, ty1 + 1, ty2] - shmem_fstar1[tx, ty1, ty2]) +
                                                inverse_weights[ty2] *
                                                (shmem_fstar2[tx, ty1, ty2 + 1] - shmem_fstar2[tx, ty1, ty2])) *
                                               (1 - dg_only)
    end

    # Compute DG volume fluxes
    for thread in 1:tile_width
        volume_flux_node1 = volume_flux_dg(u_node,
                                           get_node_vars(u, equations, thread, ty2, k),
                                           1, equations)
        volume_flux_node2 = volume_flux_dg(u_node,
                                           get_node_vars(u, equations, ty1, thread, k),
                                           2, equations)

        # Contribute DG to the volume integrals
        for tx in axes(du, 1)
            @inbounds shmem_value[tx, ty1, ty2] += (shmem_split[thread, ty1] *
                                                    (1 - isequal(ty1, thread)) * # set diagonal elements to zeros
                                                    volume_flux_node1[tx] +
                                                    shmem_split[thread, ty2] *
                                                    (1 - isequal(ty2, thread)) * # set diagonal elements to zeros
                                                    volume_flux_node2[tx]) * dg_only +
                                                   ((1 - alpha_element) * shmem_split[thread, ty1] *
                                                    (1 - isequal(ty1, thread)) * # set diagonal elements to zeros
                                                    volume_flux_node1[tx] +
                                                    (1 - alpha_element) * shmem_split[thread, ty2] *
                                                    (1 - isequal(ty2, thread)) * # set diagonal elements to zeros
                                                    volume_flux_node2[tx]) * (1 - dg_only)
        end
    end

    # Finalize the values
    for tx in axes(du, 1)
        @inbounds du[tx, ty1, ty2, k] = shmem_value[tx, ty1, ty2]
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume fluxes
function volume_flux_dgfv_kernel!(volume_flux_arr1, volume_flux_arr2, noncons_flux_arr1,
                                  noncons_flux_arr2, fstar1_L, fstar1_R, fstar2_L, fstar2_R,
                                  u, alpha, atol, derivative_split,
                                  equations::AbstractEquations{2},
                                  volume_flux_dg::Any, noncons_flux_dg::Any,
                                  volume_flux_fv::Any, noncons_flux_fv::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^3 && k <= size(u, 4))
        u2 = size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        dg_only = isapprox(alpha[k], 0, atol = atol)

        u_node = get_node_vars(u, equations, j1, j2, k)
        u_node1 = get_node_vars(u, equations, j3, j2, k)
        u_node2 = get_node_vars(u, equations, j1, j3, k)

        volume_flux_node1 = volume_flux_dg(u_node, u_node1, 1, equations)
        volume_flux_node2 = volume_flux_dg(u_node, u_node2, 2, equations)

        noncons_flux_node1 = noncons_flux_dg(u_node, u_node1, 1, equations)
        noncons_flux_node2 = noncons_flux_dg(u_node, u_node2, 2, equations)

        for ii in axes(u, 1)
            @inbounds begin
                volume_flux_arr1[ii, j1, j3, j2, k] = volume_flux_node1[ii] * derivative_split[j1, j3] *
                                                      (1 - isequal(j1, j3)) # set diagonal elements to zeros
                volume_flux_arr2[ii, j1, j2, j3, k] = volume_flux_node2[ii] * derivative_split[j2, j3] *
                                                      (1 - isequal(j2, j3)) # set diagonal elements to zeros
                noncons_flux_arr1[ii, j1, j3, j2, k] = noncons_flux_node1[ii]
                noncons_flux_arr2[ii, j1, j2, j3, k] = noncons_flux_node2[ii]
            end

            # Small optimization, no much performance gain
            if isequal(j1 + 1, j3) # avoid race condition
                f1_node = volume_flux_fv(u_node, u_node1, 1, equations)
                f1_L_node = noncons_flux_fv(u_node, u_node1, 1, equations)
                f1_R_node = noncons_flux_fv(u_node1, u_node, 1, equations)

                @inbounds begin
                    fstar1_L[ii, j3, j2, k] = f1_node[ii] + 0.5f0 * f1_L_node[ii] * (1 - dg_only)
                    fstar1_R[ii, j3, j2, k] = f1_node[ii] + 0.5f0 * f1_R_node[ii] * (1 - dg_only)
                end
            end

            if isequal(j2 + 1, j3) # avoid race condition
                f2_node = volume_flux_fv(u_node, u_node2, 2, equations)
                f2_L_node = noncons_flux_fv(u_node, u_node2, 2, equations)
                f2_R_node = noncons_flux_fv(u_node2, u_node, 2, equations)

                @inbounds begin
                    fstar2_L[ii, j1, j3, k] = f2_node[ii] + 0.5f0 * f2_L_node[ii] * (1 - dg_only)
                    fstar2_R[ii, j1, j3, k] = f2_node[ii] + 0.5f0 * f2_R_node[ii] * (1 - dg_only)
                end
            end
        end

        # if j1 != 1 && j3 == 1 # bad
        #     u_ll = get_node_vars(u, equations, j1 - 1, j2, k)
        #     u_rr = get_node_vars(u, equations, j1, j2, k)

        #     f1_node = volume_flux_fv(u_ll, u_rr, 1, equations)

        #     f1_L_node = noncons_flux_fv(u_ll, u_rr, 1, equations)
        #     f1_R_node = noncons_flux_fv(u_rr, u_ll, 1, equations)

        #     for ii in axes(u, 1)
        #         @inbounds begin
        #             fstar1_L[ii, j1, j2, k] = f1_node[ii] + 0.5f0 * f1_L_node[ii] * (1 - dg_only)
        #             fstar1_R[ii, j1, j2, k] = f1_node[ii] + 0.5f0 * f1_R_node[ii] * (1 - dg_only)
        #         end
        #     end
        # end

        # if j2 != 1 && j3 == 1 # bad
        #     u_ll = get_node_vars(u, equations, j1, j2 - 1, k)
        #     u_rr = get_node_vars(u, equations, j1, j2, k)

        #     f2_node = volume_flux_fv(u_ll, u_rr, 2, equations)

        #     f2_L_node = noncons_flux_fv(u_ll, u_rr, 2, equations)
        #     f2_R_node = noncons_flux_fv(u_rr, u_ll, 2, equations)

        #     for ii in axes(u, 1)
        #         @inbounds begin
        #             fstar2_L[ii, j1, j2, k] = f2_node[ii] + 0.5f0 * f2_L_node[ii] * (1 - dg_only)
        #             fstar2_R[ii, j1, j2, k] = f2_node[ii] + 0.5f0 * f2_R_node[ii] * (1 - dg_only)
        #         end
        #     end
        # end
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume integrals
function volume_integral_dgfv_kernel!(du, alpha, derivative_split, inverse_weights,
                                      volume_flux_arr1, volume_flux_arr2,
                                      noncons_flux_arr1, noncons_flux_arr2,
                                      fstar1_L, fstar1_R, fstar2_L, fstar2_R, atol,
                                      equations::AbstractEquations{2})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 4))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds begin
            du[i, j1, j2, k] = zero(eltype(du)) # initialize `du` with zeros
            alpha_element = alpha[k]
        end

        dg_only = isapprox(alpha_element, 0, atol = atol)

        for ii in axes(du, 2)
            @inbounds du[i, j1, j2, k] += (volume_flux_arr1[i, j1, ii, j2, k] +
                                           volume_flux_arr2[i, j1, j2, ii, k] +
                                           0.5f0 *
                                           (derivative_split[j1, ii] * noncons_flux_arr1[i, j1, ii, j2, k] +
                                            derivative_split[j2, ii] * noncons_flux_arr2[i, j1, j2, ii, k])) * dg_only +
                                          ((1 - alpha_element) *
                                           volume_flux_arr1[i, j1, ii, j2, k] +
                                           (1 - alpha_element) *
                                           volume_flux_arr2[i, j1, j2, ii, k] +
                                           0.5f0 * (1 - alpha_element) *
                                           (derivative_split[j1, ii] * noncons_flux_arr1[i, j1, ii, j2, k] +
                                            derivative_split[j2, ii] * noncons_flux_arr2[i, j1, j2, ii, k])) * (1 - dg_only)
        end

        @inbounds du[i, j1, j2, k] += alpha_element *
                                      (inverse_weights[j1] *
                                       (fstar1_L[i, j1 + 1, j2, k] - fstar1_R[i, j1, j2, k]) +
                                       inverse_weights[j2] *
                                       (fstar2_L[i, j1, j2 + 1, k] - fstar2_R[i, j1, j2, k])) * (1 - dg_only)
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating pure DG and DG-FV volume integrals with conservative terms
function volume_flux_integral_dgfv_kernel!(du, u, alpha, atol, derivative_split, inverse_weights,
                                           equations::AbstractEquations{2},
                                           volume_flux_dg::Any, noncons_flux_dg::Any,
                                           volume_flux_fv::Any, noncons_flux_fv::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_split = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_fstar1 = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width + 1, tile_width, 2), offset)
    offset += sizeof(eltype(du)) * size(du, 1) * (tile_width + 1) * tile_width * 2
    shmem_fstar2 = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width, tile_width + 1, 2), offset)
    offset += sizeof(eltype(du)) * size(du, 1) * tile_width * (tile_width + 1) * 2
    shmem_value = CuDynamicSharedArray(eltype(du), (size(du, 1), tile_width, tile_width), offset)

    # Get thread and block indices only we need save registers
    ty = threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ty1 = div(ty - 1, tile_width) + 1
    ty2 = rem(ty - 1, tile_width) + 1

    # Load global `derivative_split` into shared memory
    # Transposed load
    @inbounds shmem_split[ty1, ty2] = derivative_split[ty2, ty1]

    # Get variables for computation
    @inbounds alpha_element = alpha[k]
    dg_only = isapprox(alpha_element, 0, atol = atol)

    # Compute FV volume fluxes
    u_node = get_node_vars(u, equations, ty1, ty2, k)
    if ty1 + 1 <= tile_width
        f1_node = volume_flux_fv(u_node,
                                 get_node_vars(u, equations, ty1 + 1, ty2, k),
                                 1, equations)
        f1_L_node = noncons_flux_fv(u_node,
                                    get_node_vars(u, equations, ty1 + 1, ty2, k),
                                    1, equations)
        f1_R_node = noncons_flux_fv(get_node_vars(u, equations, ty1 + 1, ty2, k),
                                    u_node,
                                    1, equations)
    end
    if ty2 + 1 <= tile_width
        f2_node = volume_flux_fv(u_node,
                                 get_node_vars(u, equations, ty1, ty2 + 1, k),
                                 2, equations)
        f2_L_node = noncons_flux_fv(u_node,
                                    get_node_vars(u, equations, ty1, ty2 + 1, k),
                                    2, equations)
        f2_R_node = noncons_flux_fv(get_node_vars(u, equations, ty1, ty2 + 1, k),
                                    u_node,
                                    2, equations)
    end

    # Initialize the values
    for tx in axes(du, 1)
        @inbounds begin
            # Initialize `du` with zeros
            shmem_value[tx, ty1, ty2] = zero(eltype(du))

            # TODO: Remove shared memory for `fstar` and use local memory

            # Initialize `fstar` side columes with zeros (1: left)
            shmem_fstar1[tx, 1, ty2, 1] = zero(eltype(du))
            shmem_fstar1[tx, tile_width + 1, ty2, 1] = zero(eltype(du))
            shmem_fstar2[tx, ty1, 1, 1] = zero(eltype(du))
            shmem_fstar2[tx, ty1, tile_width + 1, 1] = zero(eltype(du))

            # Initialize `fstar` side columes with zeros (2: right)
            shmem_fstar1[tx, 1, ty2, 2] = zero(eltype(du))
            shmem_fstar1[tx, tile_width + 1, ty2, 2] = zero(eltype(du))
            shmem_fstar2[tx, ty1, 1, 2] = zero(eltype(du))
            shmem_fstar2[tx, ty1, tile_width + 1, 2] = zero(eltype(du))
        end

        if ty1 + 1 <= tile_width
            # Set with FV volume fluxes
            @inbounds begin
                shmem_fstar1[tx, ty1 + 1, ty2, 1] = f1_node[tx] + 0.5f0 * f1_L_node[tx] * (1 - dg_only)
                shmem_fstar1[tx, ty1 + 1, ty2, 2] = f1_node[tx] + 0.5f0 * f1_R_node[tx] * (1 - dg_only)
            end
        end
        if ty2 + 1 <= tile_width
            # Set with FV volume fluxes
            @inbounds begin
                shmem_fstar2[tx, ty1, ty2 + 1, 1] = f2_node[tx] + 0.5f0 * f2_L_node[tx] * (1 - dg_only)
                shmem_fstar2[tx, ty1, ty2 + 1, 2] = f2_node[tx] + 0.5f0 * f2_R_node[tx] * (1 - dg_only)
            end
        end
    end

    sync_threads()

    # Contribute FV to the volume integrals
    for tx in axes(du, 1)
        @inbounds shmem_value[tx, ty1, ty2] += alpha_element *
                                               (inverse_weights[ty1] *
                                                (shmem_fstar1[tx, ty1 + 1, ty2, 1] - shmem_fstar1[tx, ty1, ty2, 2]) +
                                                inverse_weights[ty2] *
                                                (shmem_fstar2[tx, ty1, ty2 + 1, 1] - shmem_fstar2[tx, ty1, ty2, 2])) * (1 - dg_only)
    end

    # Compute DG volume fluxes
    for thread in 1:tile_width
        volume_flux_node1 = volume_flux_dg(u_node,
                                           get_node_vars(u, equations, thread, ty2, k),
                                           1, equations)
        volume_flux_node2 = volume_flux_dg(u_node,
                                           get_node_vars(u, equations, ty1, thread, k),
                                           2, equations)

        noncons_flux_node1 = noncons_flux_dg(u_node,
                                             get_node_vars(u, equations, thread, ty2, k),
                                             1, equations)
        noncons_flux_node2 = noncons_flux_dg(u_node,
                                             get_node_vars(u, equations, ty1, thread, k),
                                             2, equations)

        # Contribute DG to the volume integrals
        for tx in axes(du, 1)
            @inbounds shmem_value[tx, ty1, ty2] += (volume_flux_node1[tx] * shmem_split[thread, ty1] *
                                                    (1 - isequal(ty1, thread)) + # set diagonal elements to zeros
                                                    volume_flux_node2[tx] * shmem_split[thread, ty2] *
                                                    (1 - isequal(ty2, thread)) +
                                                    0.5f0 *
                                                    (shmem_split[thread, ty1] * noncons_flux_node1[tx] +
                                                     shmem_split[thread, ty2] * noncons_flux_node2[tx])) * dg_only +
                                                   ((1 - alpha_element) *
                                                    volume_flux_node1[tx] * shmem_split[thread, ty1] *
                                                    (1 - isequal(ty1, thread)) + # set diagonal elements to zeros
                                                    (1 - alpha_element) *
                                                    volume_flux_node2[tx] * shmem_split[thread, ty2] *
                                                    (1 - isequal(ty2, thread)) + # set diagonal elements to zeros
                                                    0.5f0 * (1 - alpha_element) *
                                                    (shmem_split[thread, ty1] * noncons_flux_node1[tx] +
                                                     shmem_split[thread, ty2] * noncons_flux_node2[tx])) * (1 - dg_only)
        end
    end

    # Finalize the values
    for tx in axes(du, 1)
        @inbounds du[tx, ty1, ty2, k] = shmem_value[tx, ty1, ty2]
    end

    return nothing
end

# Kernel for prolonging two interfaces 
function prolong_interfaces_kernel!(interfaces_u, u, neighbor_ids, orientations,
                                    euqations::AbstractEquations{2})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(interfaces_u, 2) * size(interfaces_u, 3) && k <= size(interfaces_u, 4))
        u2 = size(u, 2) # size(interfaces_u, 3) == size(u, 2)

        j1 = div(j - 1, u2) + 1
        j2 = rem(j - 1, u2) + 1

        @inbounds begin
            orientation = orientations[k]
            left_element = neighbor_ids[1, k]
            right_element = neighbor_ids[2, k]

            interfaces_u[1, j1, j2, k] = u[j1,
                                           (2 - orientation) * u2 + (orientation - 1) * j2,
                                           (2 - orientation) * j2 + (orientation - 1) * u2,
                                           left_element]
            interfaces_u[2, j1, j2, k] = u[j1,
                                           (2 - orientation) + (orientation - 1) * j2,
                                           (2 - orientation) * j2 + (orientation - 1),
                                           right_element]
        end
    end

    return nothing
end

# Kernel for calculating surface fluxes 
function surface_flux_kernel!(surface_flux_arr, interfaces_u, orientations,
                              equations::AbstractEquations{2}, surface_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(surface_flux_arr, 2) && k <= size(surface_flux_arr, 3))
        u_ll, u_rr = get_surface_node_vars(interfaces_u, equations, j, k)
        @inbounds orientation = orientations[k]

        surface_flux_node = surface_flux(u_ll, u_rr, orientation, equations)

        for ii in axes(surface_flux_arr, 1)
            @inbounds surface_flux_arr[ii, j, k] = surface_flux_node[ii]
        end
    end

    return nothing
end

# Kernel for calculating surface and both nonconservative fluxes 
function surface_noncons_flux_kernel!(surface_flux_arr, noncons_left_arr, noncons_right_arr,
                                      interfaces_u, orientations, equations::AbstractEquations{2},
                                      surface_flux::Any, nonconservative_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(surface_flux_arr, 2) && k <= size(surface_flux_arr, 3))
        u_ll, u_rr = get_surface_node_vars(interfaces_u, equations, j, k)
        @inbounds orientation = orientations[k]

        surface_flux_node = surface_flux(u_ll, u_rr, orientation, equations)
        noncons_left_node = nonconservative_flux(u_ll, u_rr, orientation, equations)
        noncons_right_node = nonconservative_flux(u_rr, u_ll, orientation, equations)

        for ii in axes(surface_flux_arr, 1)
            @inbounds begin
                surface_flux_arr[ii, j, k] = surface_flux_node[ii]
                noncons_left_arr[ii, j, k] = noncons_left_node[ii]
                noncons_right_arr[ii, j, k] = noncons_right_node[ii]
            end
        end
    end

    return nothing
end

# Kernel for setting interface fluxes
function interface_flux_kernel!(surface_flux_values, surface_flux_arr, neighbor_ids, orientations,
                                equations::AbstractEquations{2})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_arr, 2) &&
        k <= size(surface_flux_arr, 3))
        @inbounds begin
            left_id = neighbor_ids[1, k]
            right_id = neighbor_ids[2, k]

            left_direction = 2 * orientations[k]
            right_direction = 2 * orientations[k] - 1

            surface_flux_values[i, j, left_direction, left_id] = surface_flux_arr[i, j, k]
            surface_flux_values[i, j, right_direction, right_id] = surface_flux_arr[i, j, k]
        end
    end

    return nothing
end

# Kernel for setting interface fluxes
function interface_flux_kernel!(surface_flux_values, surface_flux_arr, noncons_left_arr,
                                noncons_right_arr, neighbor_ids, orientations,
                                equations::AbstractEquations{2})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_arr, 2) &&
        k <= size(surface_flux_arr, 3))
        @inbounds begin
            left_id = neighbor_ids[1, k]
            right_id = neighbor_ids[2, k]

            left_direction = 2 * orientations[k]
            right_direction = 2 * orientations[k] - 1

            surface_flux_values[i, j, left_direction, left_id] = surface_flux_arr[i, j, k] +
                                                                 0.5f0 *
                                                                 noncons_left_arr[i, j, k]
            surface_flux_values[i, j, right_direction, right_id] = surface_flux_arr[i, j, k] +
                                                                   0.5f0 *
                                                                   noncons_right_arr[i, j, k]
        end
    end

    return nothing
end

# Kernel for prolonging two boundaries
function prolong_boundaries_kernel!(boundaries_u, u, neighbor_ids, neighbor_sides, orientations,
                                    equations::AbstractEquations{2})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(boundaries_u, 2) * size(boundaries_u, 3) && k <= size(boundaries_u, 4))
        u2 = size(u, 2) # size(boundaries_u, 3) == size(u, 2)

        j1 = div(j - 1, u2) + 1
        j2 = rem(j - 1, u2) + 1

        @inbounds begin
            element = neighbor_ids[k]
            side = neighbor_sides[k]
            orientation = orientations[k]

            boundaries_u[1, j1, j2, k] = u[j1,
                                           (2 - orientation) * u2 + (orientation - 1) * j2,
                                           (2 - orientation) * j2 + (orientation - 1) * u2,
                                           element] * (2 - side) # Set to 0 instead of NaN
            boundaries_u[2, j1, j2, k] = u[j1,
                                           (2 - orientation) + (orientation - 1) * j2,
                                           (2 - orientation) * j2 + (orientation - 1),
                                           element] * (side - 1) # Set to 0 instead of NaN
        end
    end

    return nothing
end

# Kernel for calculating boundary fluxes
function boundary_flux_kernel!(surface_flux_values, boundaries_u, node_coordinates, t, boundary_arr,
                               indices_arr, neighbor_ids, neighbor_sides, orientations,
                               boundary_conditions::NamedTuple, equations::AbstractEquations{2},
                               surface_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(surface_flux_values, 2) && k <= length(boundary_arr))
        @inbounds begin
            boundary = boundary_arr[k]
            direction = (indices_arr[1] <= boundary) + (indices_arr[2] <= boundary) +
                        (indices_arr[3] <= boundary) + (indices_arr[4] <= boundary)

            neighbor = neighbor_ids[boundary]
            side = neighbor_sides[boundary]
            orientation = orientations[boundary]
        end

        u_ll, u_rr = get_surface_node_vars(boundaries_u, equations, j, boundary)
        u_inner = (2 - side) * u_ll + (side - 1) * u_rr
        x = get_node_coords(node_coordinates, equations, j, boundary)

        # TODO: Improve this part
        if direction == 1
            boundary_flux_node = boundary_conditions[1](u_inner, orientation,
                                                        direction, x, t, surface_flux, equations)
        elseif direction == 2
            boundary_flux_node = boundary_conditions[2](u_inner, orientation,
                                                        direction, x, t, surface_flux, equations)
        elseif direction == 3
            boundary_flux_node = boundary_conditions[3](u_inner, orientation,
                                                        direction, x, t, surface_flux, equations)
        else
            boundary_flux_node = boundary_conditions[4](u_inner, orientation,
                                                        direction, x, t, surface_flux, equations)
        end

        for ii in axes(surface_flux_values, 1)
            # `boundary_flux_node` can be nothing if periodic boundary condition is applied
            @inbounds surface_flux_values[ii, j, direction, neighbor] = isnothing(boundary_flux_node) ? # bad
                                                                        surface_flux_values[ii, j,
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
                               boundary_conditions::NamedTuple, equations::AbstractEquations{2},
                               surface_flux::Any, nonconservative_terms::True)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(surface_flux_values, 2) && k <= length(boundary_arr))
        @inbounds begin
            boundary = boundary_arr[k]
            direction = (indices_arr[1] <= boundary) + (indices_arr[2] <= boundary) +
                        (indices_arr[3] <= boundary) + (indices_arr[4] <= boundary)

            neighbor = neighbor_ids[boundary]
            side = neighbor_sides[boundary]
            orientation = orientations[boundary]
        end

        u_ll, u_rr = get_surface_node_vars(boundaries_u, equations, j, boundary)
        u_inner = (2 - side) * u_ll + (side - 1) * u_rr
        x = get_node_coords(node_coordinates, equations, j, boundary)

        # TODO: Improve this part
        if direction == 1
            flux_node, noncons_flux_node = boundary_conditions[1](u_inner, orientation, direction,
                                                                  x, t, surface_flux, equations)
        elseif direction == 2
            flux_node, noncons_flux_node = boundary_conditions[2](u_inner, orientation, direction,
                                                                  x, t, surface_flux, equations)
        elseif direction == 3
            flux_node, noncons_flux_node = boundary_conditions[3](u_inner, orientation, direction,
                                                                  x, t, surface_flux, equations)
        else
            flux_node, noncons_flux_node = boundary_conditions[4](u_inner, orientation, direction,
                                                                  x, t, surface_flux, equations)
        end

        for ii in axes(surface_flux_values, 1)
            @inbounds surface_flux_values[ii, j, direction, neighbor] = flux_node[ii] +
                                                                        0.5f0 * noncons_flux_node[ii]
        end
    end

    return nothing
end

# Kernel for copying data small to small on mortars
function prolong_mortars_small2small_kernel!(u_upper, u_lower, u, neighbor_ids, large_sides,
                                             orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u_upper, 2) && j <= size(u_upper, 3) && k <= size(u_upper, 4))
        @inbounds begin
            large_side = large_sides[k]
            orientation = orientations[k]

            lower_element = neighbor_ids[1, k]
            upper_element = neighbor_ids[2, k]
        end

        u2 = size(u, 2)

        @inbounds begin
            u_upper[2, i, j, k] = u[i,
                                    (2 - orientation) + (orientation - 1) * j,
                                    (2 - orientation) * j + (orientation - 1),
                                    upper_element] * (2 - large_side)

            u_lower[2, i, j, k] = u[i,
                                    (2 - orientation) + (orientation - 1) * j,
                                    (2 - orientation) * j + (orientation - 1),
                                    lower_element] * (2 - large_side)

            u_upper[1, i, j, k] = u[i,
                                    (2 - orientation) * u2 + (orientation - 1) * j,
                                    (2 - orientation) * j + (orientation - 1) * u2,
                                    upper_element] * (large_side - 1)

            u_lower[1, i, j, k] = u[i,
                                    (2 - orientation) * u2 + (orientation - 1) * j,
                                    (2 - orientation) * j + (orientation - 1) * u2,
                                    lower_element] * (large_side - 1)
        end
    end

    return nothing
end

# Kernel for interpolating data large to small on mortars
function prolong_mortars_large2small_kernel!(u_upper, u_lower, u, forward_upper, forward_lower,
                                             neighbor_ids, large_sides, orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u_upper, 2) && j <= size(u_upper, 3) && k <= size(u_upper, 4))
        @inbounds begin
            large_side = large_sides[k]
            orientation = orientations[k]
            large_element = neighbor_ids[3, k]
        end

        leftright = large_side
        u2 = size(u, 2)

        for jj in axes(forward_upper, 2)
            @inbounds begin
                u_upper[leftright, i, j, k] += forward_upper[j, jj] *
                                               u[i,
                                                 (2 - orientation) * u2 + (orientation - 1) * jj,
                                                 (2 - orientation) * jj + (orientation - 1) * u2,
                                                 large_element] * (2 - large_side)
                u_lower[leftright, i, j, k] += forward_lower[j, jj] *
                                               u[i,
                                                 (2 - orientation) * u2 + (orientation - 1) * jj,
                                                 (2 - orientation) * jj + (orientation - 1) * u2,
                                                 large_element] * (2 - large_side)
            end
        end

        for jj in axes(forward_lower, 2)
            @inbounds begin
                u_upper[leftright, i, j, k] += forward_upper[j, jj] *
                                               u[i,
                                                 (2 - orientation) + (orientation - 1) * jj,
                                                 (2 - orientation) * jj + (orientation - 1),
                                                 large_element] * (large_side - 1)
                u_lower[leftright, i, j, k] += forward_lower[j, jj] *
                                               u[i,
                                                 (2 - orientation) + (orientation - 1) * jj,
                                                 (2 - orientation) * jj + (orientation - 1),
                                                 large_element] * (large_side - 1)
            end
        end
    end

    return nothing
end

# Kernel for calculating mortar fluxes
function mortar_flux_kernel!(fstar_primary_upper, fstar_primary_lower, fstar_secondary_upper,
                             fstar_secondary_lower, u_upper, u_lower, orientations,
                             equations::AbstractEquations{2}, surface_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u_upper, 3) && k <= length(orientations))
        u_upper_ll, u_upper_rr = get_surface_node_vars(u_upper, equations, j, k)
        u_lower_ll, u_lower_rr = get_surface_node_vars(u_lower, equations, j, k)
        @inbounds orientation = orientations[k]

        flux_upper_node = surface_flux(u_upper_ll, u_upper_rr, orientation, equations)
        flux_lower_node = surface_flux(u_lower_ll, u_lower_rr, orientation, equations)

        for ii in axes(fstar_primary_upper, 1)
            @inbounds begin
                fstar_primary_upper[ii, j, k] = flux_upper_node[ii]
                fstar_primary_lower[ii, j, k] = flux_lower_node[ii]
                fstar_secondary_upper[ii, j, k] = flux_upper_node[ii]
                fstar_secondary_lower[ii, j, k] = flux_lower_node[ii]
            end
        end
    end

    return nothing
end

# Kernel for calculating mortar fluxes and adding nonconservative fluxes
function mortar_flux_kernel!(fstar_primary_upper, fstar_primary_lower, fstar_secondary_upper,
                             fstar_secondary_lower, u_upper, u_lower, orientations, large_sides,
                             equations::AbstractEquations{2}, surface_flux::Any,
                             nonconservative_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u_upper, 3) && k <= length(orientations))
        u_upper_ll, u_upper_rr = get_surface_node_vars(u_upper, equations, j, k)
        u_lower_ll, u_lower_rr = get_surface_node_vars(u_lower, equations, j, k)

        @inbounds begin
            orientation = orientations[k]
            large_side = large_sides[k]
        end

        flux_upper_node = surface_flux(u_upper_ll, u_upper_rr, orientation, equations)
        flux_lower_node = surface_flux(u_lower_ll, u_lower_rr, orientation, equations)

        for ii in axes(fstar_primary_upper, 1)
            @inbounds begin
                fstar_primary_upper[ii, j, k] = flux_upper_node[ii]
                fstar_primary_lower[ii, j, k] = flux_lower_node[ii]
                fstar_secondary_upper[ii, j, k] = flux_upper_node[ii]
                fstar_secondary_lower[ii, j, k] = flux_lower_node[ii]
            end
        end

        u_upper1 = (2 - large_side) * u_upper_ll + (large_side - 1) * u_upper_rr
        u_upper2 = (large_side - 1) * u_upper_ll + (2 - large_side) * u_upper_rr

        u_lower1 = (2 - large_side) * u_lower_ll + (large_side - 1) * u_lower_rr
        u_lower2 = (large_side - 1) * u_lower_ll + (2 - large_side) * u_lower_rr

        noncons_flux_primary_upper = nonconservative_flux(u_upper1, u_upper2, orientation,
                                                          equations)
        noncons_flux_primary_lower = nonconservative_flux(u_lower1, u_lower2, orientation,
                                                          equations)
        noncons_flux_secondary_upper = nonconservative_flux(u_upper2, u_upper1, orientation,
                                                            equations)
        noncons_flux_secondary_lower = nonconservative_flux(u_lower2, u_lower1, orientation,
                                                            equations)

        for ii in axes(fstar_primary_upper, 1)
            @inbounds begin
                fstar_primary_upper[ii, j, k] += 0.5f0 * noncons_flux_primary_upper[ii]
                fstar_primary_lower[ii, j, k] += 0.5f0 * noncons_flux_primary_lower[ii]
                fstar_secondary_upper[ii, j, k] += 0.5f0 * noncons_flux_secondary_upper[ii]
                fstar_secondary_lower[ii, j, k] += 0.5f0 * noncons_flux_secondary_lower[ii]
            end
        end
    end

    return nothing
end

# Kernel for copying mortar fluxes small to small and small to large
function mortar_flux_copy_to_kernel!(surface_flux_values, tmp_surface_flux_values,
                                     fstar_primary_upper, fstar_primary_lower,
                                     fstar_secondary_upper, fstar_secondary_lower,
                                     reverse_upper, reverse_lower, neighbor_ids, large_sides,
                                     orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_values, 2) &&
        k <= length(orientations))
        @inbounds begin
            large_element = neighbor_ids[3, k]
            upper_element = neighbor_ids[2, k]
            lower_element = neighbor_ids[1, k]

            large_side = large_sides[k]
            orientation = orientations[k]

            # Use math expression to enhance performance (against control flow), it is equivalent to,
            # `(2 - large_side) * (2 - orientation) * 1 + 
            #  (2 - large_side) * (orientation - 1) * 3 +
            #  (large_side - 1) * (2 - orientation) * 2 +
            #  (large_side - 1) * (orientation - 1) * 4`.
            direction = large_side + 2 * orientation - 2

            surface_flux_values[i, j, direction, upper_element] = fstar_primary_upper[i, j, k]
            surface_flux_values[i, j, direction, lower_element] = fstar_primary_lower[i, j, k]

            # Use math expression to enhance performance (against control flow), it is equivalent to,
            # `(2 - large_side) * (2 - orientation) * 2 + 
            #  (2 - large_side) * (orientation - 1) * 4 +
            #  (large_side - 1) * (2 - orientation) * 1 +
            #  (large_side - 1) * (orientation - 1) * 3`.
            direction = 2 * orientation - large_side + 1
        end

        for ii in axes(reverse_upper, 2) # i.e., ` for ii in axes(reverse_lower, 2)`
            @inbounds tmp_surface_flux_values[i, j, direction, large_element] += fstar_secondary_upper[i, ii, k] *
                                                                                 reverse_upper[j, ii] +
                                                                                 fstar_secondary_lower[i, ii, k] *
                                                                                 reverse_lower[j, ii]
        end

        @inbounds surface_flux_values[i, j, direction, large_element] = tmp_surface_flux_values[i, j,
                                                                                                direction,
                                                                                                large_element]
    end

    return nothing
end

# Kernel for calculating surface integrals
function surface_integral_kernel!(du, factor_arr, surface_flux_values,
                                  equations::AbstractEquations{2})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 4))
        u2 = size(du, 2)

        j1 = div(j - 1, u2) + 1
        j2 = rem(j - 1, u2) + 1

        @inbounds begin
            du[i, j1, j2, k] -= (surface_flux_values[i, j2, 1, k] * isequal(j1, 1) +
                                 surface_flux_values[i, j1, 3, k] * isequal(j2, 1)) * factor_arr[1]
            du[i, j1, j2, k] += (surface_flux_values[i, j2, 2, k] * isequal(j1, u2) +
                                 surface_flux_values[i, j1, 4, k] * isequal(j2, u2)) * factor_arr[2]
        end
    end

    return nothing
end

# Kernel for applying inverse Jacobian 
function jacobian_kernel!(du, inverse_jacobian, equations::AbstractEquations{2})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 4))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds du[i, j1, j2, k] *= -inverse_jacobian[k]
    end

    return nothing
end

# CUDA kernel for calculating source terms
function source_terms_kernel!(du, u, node_coordinates, t, equations::AbstractEquations{2},
                              source_terms::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(du, 2)^2 && k <= size(du, 4))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        u_local = get_node_vars(u, equations, j1, j2, k)
        x_local = get_node_coords(node_coordinates, equations, j1, j2, k)

        source_terms_node = source_terms(u_local, x_local, t, equations)

        for ii in axes(du, 1)
            @inbounds du[ii, j1, j2, k] += source_terms_node[ii]
        end
    end

    return nothing
end
