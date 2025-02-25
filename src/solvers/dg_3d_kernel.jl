# GPU kernels related to a DG semidiscretization in 3D.

# Functions that end with `_kernel` are CUDA kernels that are going to be launched by 
# the @cuda macro with parameters from the kernel configurator. They are purely run on 
# the device (i.e., GPU).

# Kernel for calculating fluxes along normal directions
function flux_kernel!(flux_arr1, flux_arr2, flux_arr3, u, equations::AbstractEquations{3},
                      flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^3 && k <= size(u, 5))
        u2 = size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        u_node = get_node_vars(u, equations, j1, j2, j3, k)

        flux_node1 = flux(u_node, 1, equations)
        flux_node2 = flux(u_node, 2, equations)
        flux_node3 = flux(u_node, 3, equations)

        for ii in axes(u, 1)
            @inbounds begin
                flux_arr1[ii, j1, j2, j3, k] = flux_node1[ii]
                flux_arr2[ii, j1, j2, j3, k] = flux_node2[ii]
                flux_arr3[ii, j1, j2, j3, k] = flux_node3[ii]
            end
        end
    end

    return nothing
end

# Kernel for calculating weak form
function weak_form_kernel!(du, derivative_dhat, flux_arr1, flux_arr2, flux_arr3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^3 && k <= size(du, 5))
        u2 = size(du, 2) # size(du, 2) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        @inbounds du[i, j1, j2, j3, k] = zero(eltype(du)) # initialize `du` with zeros

        for ii in axes(du, 2)
            @inbounds du[i, j1, j2, j3, k] += derivative_dhat[j1, ii] * flux_arr1[i, ii, j2, j3, k] +
                                              derivative_dhat[j2, ii] * flux_arr2[i, j1, ii, j3, k] +
                                              derivative_dhat[j3, ii] * flux_arr3[i, j1, j2, ii, k]
        end
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating volume integrals with weak form
function flux_weak_form_kernel!(du, u, derivative_dhat,
                                equations::AbstractEquations{3}, flux::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_dhat = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_flux = CuDynamicSharedArray(eltype(du),
                                      (size(du, 1), tile_width, tile_width, tile_width, 3), offset)

    # Get thread and block indices only we need save registers
    tx, ty = threadIdx().x, threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ty1 = div(ty - 1, tile_width^2) + 1
    ty2 = div(rem(ty - 1, tile_width^2), tile_width) + 1
    ty3 = rem(rem(ty - 1, tile_width^2), tile_width) + 1

    # Tile the computation (restrict to one tile here)
    value = zero(eltype(du))

    # Load global `derivative_dhat` into shared memory
    # Transposed load
    @inbounds shmem_dhat[ty1, ty2] = derivative_dhat[ty2, ty1]

    # Compute flux values
    u_node = get_node_vars(u, equations, ty1, ty2, ty3, k)
    flux_node1 = flux(u_node, 1, equations)
    flux_node2 = flux(u_node, 2, equations)
    flux_node3 = flux(u_node, 3, equations)

    @inbounds begin
        shmem_flux[tx, ty1, ty2, ty3, 1] = flux_node1[tx]
        shmem_flux[tx, ty1, ty2, ty3, 2] = flux_node2[tx]
        shmem_flux[tx, ty1, ty2, ty3, 3] = flux_node3[tx]
    end

    sync_threads()

    # Loop within one block to get weak form
    # TODO: Avoid potential bank conflicts
    for thread in 1:tile_width
        @inbounds value += shmem_dhat[thread, ty1] * shmem_flux[tx, thread, ty2, ty3, 1] +
                           shmem_dhat[thread, ty2] * shmem_flux[tx, ty1, thread, ty3, 2] +
                           shmem_dhat[thread, ty3] * shmem_flux[tx, ty1, ty2, thread, 3]
    end

    # Synchronization is not needed here if we use only one tile
    # sync_threads()

    # Finalize the weak form
    @inbounds du[tx, ty1, ty2, ty3, k] = value

    return nothing
end

# CUDA kernel for calculating volume fluxes
function volume_flux_kernel!(volume_flux_arr1, volume_flux_arr2, volume_flux_arr3, u,
                             equations::AbstractEquations{3}, volume_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^4 && k <= size(u, 5))
        u2 = size(u, 2)

        j1 = div(j - 1, u2^3) + 1
        j2 = div(rem(j - 1, u2^3), u2^2) + 1
        j3 = div(rem(j - 1, u2^2), u2) + 1
        j4 = rem(j - 1, u2) + 1

        u_node = get_node_vars(u, equations, j1, j2, j3, k)
        u_node1 = get_node_vars(u, equations, j4, j2, j3, k)
        u_node2 = get_node_vars(u, equations, j1, j4, j3, k)
        u_node3 = get_node_vars(u, equations, j1, j2, j4, k)

        volume_flux_node1 = volume_flux(u_node, u_node1, 1, equations)
        volume_flux_node2 = volume_flux(u_node, u_node2, 2, equations)
        volume_flux_node3 = volume_flux(u_node, u_node3, 3, equations)

        for ii in axes(u, 1)
            @inbounds begin
                volume_flux_arr1[ii, j1, j4, j2, j3, k] = volume_flux_node1[ii]
                volume_flux_arr2[ii, j1, j2, j4, j3, k] = volume_flux_node2[ii]
                volume_flux_arr3[ii, j1, j2, j3, j4, k] = volume_flux_node3[ii]
            end
        end
    end

    return nothing
end

# Kernel for calculating volume integrals
function volume_integral_kernel!(du, derivative_split, volume_flux_arr1, volume_flux_arr2,
                                 volume_flux_arr3, equations::AbstractEquations{3})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^3 && k <= size(du, 5))
        u2 = size(du, 2) # size(du, 2) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        @inbounds du[i, j1, j2, j3, k] = zero(eltype(du)) # initialize `du` with zeros

        for ii in axes(du, 2)
            @inbounds du[i, j1, j2, j3, k] += volume_flux_arr1[i, j1, ii, j2, j3, k] * derivative_split[j1, ii] *
                                              (1 - isequal(j1, ii)) + # set diagonal elements to zeros
                                              volume_flux_arr2[i, j1, j2, ii, j3, k] * derivative_split[j2, ii] *
                                              (1 - isequal(j2, ii)) + # set diagonal elements to zeros
                                              volume_flux_arr3[i, j1, j2, j3, ii, k] * derivative_split[j3, ii] *
                                              (1 - isequal(j3, ii)) # set diagonal elements to zeros
        end
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating volume integrals without conservative terms
function volume_flux_integral_kernel!(du, u, derivative_split,
                                      equations::AbstractEquations{3}, volume_flux::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_split = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_value = CuDynamicSharedArray(eltype(du),
                                       (size(du, 1), tile_width, tile_width, tile_width), offset)

    # Get thread and block indices only we need save registers
    ty = threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ty1 = div(ty - 1, tile_width^2) + 1
    ty2 = div(rem(ty - 1, tile_width^2), tile_width) + 1
    ty3 = rem(rem(ty - 1, tile_width^2), tile_width) + 1

    # Tile the computation (set to one tile here)
    # Initialize the values
    for tx in axes(du, 1)
        @inbounds shmem_value[tx, ty1, ty2, ty3] = zero(eltype(du))
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
        u_node = get_node_vars(u, equations, ty1, ty2, ty3, k)
        volume_flux_node1 = volume_flux(u_node,
                                        get_node_vars(u, equations, thread, ty2, ty3, k),
                                        1, equations)
        volume_flux_node2 = volume_flux(u_node,
                                        get_node_vars(u, equations, ty1, thread, ty3, k),
                                        2, equations)
        volume_flux_node3 = volume_flux(u_node,
                                        get_node_vars(u, equations, ty1, ty2, thread, k),
                                        3, equations)

        # TODO: Avoid potential bank conflicts 
        # Try another way to parallelize (ty1, ty2, ty3) with threads to ty4, 
        # then consolidate each computation back to (ty1, ty2, ty3)
        for tx in axes(du, 1)
            @inbounds shmem_value[tx, ty1, ty2, ty3] += shmem_split[thread, ty1] * volume_flux_node1[tx] +
                                                        shmem_split[thread, ty2] * volume_flux_node2[tx] +
                                                        shmem_split[thread, ty3] * volume_flux_node3[tx]
        end
    end

    # Synchronization is not needed here if we use only one tile
    # sync_threads()

    # Finalize the values
    for tx in axes(du, 1)
        @inbounds du[tx, ty1, ty2, ty3, k] = shmem_value[tx, ty1, ty2, ty3]
    end

    return nothing
end

# Kernel for calculating symmetric and nonconservative fluxes
function noncons_volume_flux_kernel!(symmetric_flux_arr1, symmetric_flux_arr2, symmetric_flux_arr3,
                                     noncons_flux_arr1, noncons_flux_arr2, noncons_flux_arr3,
                                     u, derivative_split, equations::AbstractEquations{3},
                                     symmetric_flux::Any, nonconservative_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^4 && k <= size(u, 5))
        u2 = size(u, 2)

        j1 = div(j - 1, u2^3) + 1
        j2 = div(rem(j - 1, u2^3), u2^2) + 1
        j3 = div(rem(j - 1, u2^2), u2) + 1
        j4 = rem(j - 1, u2) + 1

        u_node = get_node_vars(u, equations, j1, j2, j3, k)
        u_node1 = get_node_vars(u, equations, j4, j2, j3, k)
        u_node2 = get_node_vars(u, equations, j1, j4, j3, k)
        u_node3 = get_node_vars(u, equations, j1, j2, j4, k)

        symmetric_flux_node1 = symmetric_flux(u_node, u_node1, 1, equations)
        symmetric_flux_node2 = symmetric_flux(u_node, u_node2, 2, equations)
        symmetric_flux_node3 = symmetric_flux(u_node, u_node3, 3, equations)

        noncons_flux_node1 = nonconservative_flux(u_node, u_node1, 1, equations)
        noncons_flux_node2 = nonconservative_flux(u_node, u_node2, 2, equations)
        noncons_flux_node3 = nonconservative_flux(u_node, u_node3, 3, equations)

        for ii in axes(u, 1)
            @inbounds begin
                symmetric_flux_arr1[ii, j1, j4, j2, j3, k] = symmetric_flux_node1[ii] * derivative_split[j1, j4] *
                                                             (1 - isequal(j1, j4)) # set diagonal elements to zeros      
                symmetric_flux_arr2[ii, j1, j2, j4, j3, k] = symmetric_flux_node2[ii] * derivative_split[j2, j4] *
                                                             (1 - isequal(j2, j4)) # set diagonal elements to zeros
                symmetric_flux_arr3[ii, j1, j2, j3, j4, k] = symmetric_flux_node3[ii] * derivative_split[j3, j4] *
                                                             (1 - isequal(j3, j4)) # set diagonal elements to zeros

                noncons_flux_arr1[ii, j1, j4, j2, j3, k] = noncons_flux_node1[ii]
                noncons_flux_arr2[ii, j1, j2, j4, j3, k] = noncons_flux_node2[ii]
                noncons_flux_arr3[ii, j1, j2, j3, j4, k] = noncons_flux_node3[ii]
            end
        end
    end

    return nothing
end

# Kernel for calculating symmetric and nonconservative volume integrals
function volume_integral_kernel!(du, derivative_split,
                                 symmetric_flux_arr1, symmetric_flux_arr2, symmetric_flux_arr3,
                                 noncons_flux_arr1, noncons_flux_arr2, noncons_flux_arr3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^3 && k <= size(du, 5))
        u2 = size(du, 2) # size(du, 2) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        @inbounds du[i, j1, j2, j3, k] = zero(eltype(du)) # initialize `du` with zeros

        for ii in axes(du, 2)
            @inbounds du[i, j1, j2, j3, k] += symmetric_flux_arr1[i, j1, ii, j2, j3, k] +
                                              symmetric_flux_arr2[i, j1, j2, ii, j3, k] +
                                              symmetric_flux_arr3[i, j1, j2, j3, ii, k] +
                                              0.5f0 *
                                              derivative_split[j1, ii] * noncons_flux_arr1[i, j1, ii, j2, j3, k] +
                                              0.5f0 *
                                              derivative_split[j2, ii] * noncons_flux_arr2[i, j1, j2, ii, j3, k] +
                                              0.5f0 *
                                              derivative_split[j3, ii] * noncons_flux_arr3[i, j1, j2, j3, ii, k]
        end
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating volume integrals with conservative terms
function volume_flux_integral_kernel!(du, u, derivative_split,
                                      equations::AbstractEquations{3},
                                      symmetric_flux::Any, nonconservative_flux::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_split = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_value = CuDynamicSharedArray(eltype(du),
                                       (size(du, 1), tile_width, tile_width, tile_width), offset)

    # Get thread and block indices only we need save registers
    ty = threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ty1 = div(ty - 1, tile_width^2) + 1
    ty2 = div(rem(ty - 1, tile_width^2), tile_width) + 1
    ty3 = rem(rem(ty - 1, tile_width^2), tile_width) + 1

    # Tile the computation (set to one tile here)
    # Initialize the values
    for tx in axes(du, 1)
        @inbounds shmem_value[tx, ty1, ty2, ty3] = zero(eltype(du))
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
        u_node = get_node_vars(u, equations, ty1, ty2, ty3, k)
        symmetric_flux_node1 = symmetric_flux(u_node,
                                              get_node_vars(u, equations, thread, ty2, ty3, k),
                                              1, equations)
        symmetric_flux_node2 = symmetric_flux(u_node,
                                              get_node_vars(u, equations, ty1, thread, ty3, k),
                                              2, equations)
        symmetric_flux_node3 = symmetric_flux(u_node,
                                              get_node_vars(u, equations, ty1, ty2, thread, k),
                                              3, equations)
        noncons_flux_node1 = nonconservative_flux(u_node,
                                                  get_node_vars(u, equations, thread, ty2, ty3, k),
                                                  1, equations)
        noncons_flux_node2 = nonconservative_flux(u_node,
                                                  get_node_vars(u, equations, ty1, thread, ty3, k),
                                                  2, equations)
        noncons_flux_node3 = nonconservative_flux(u_node,
                                                  get_node_vars(u, equations, ty1, ty2, thread, k),
                                                  3, equations)

        # TODO: Avoid potential bank conflicts
        for tx in axes(du, 1)
            @inbounds shmem_value[tx, ty1, ty2, ty3] += symmetric_flux_node1[tx] * shmem_split[thread, ty1] *
                                                        (1 - isequal(ty1, thread)) + # set diagonal elements to zeros
                                                        symmetric_flux_node2[tx] * shmem_split[thread, ty2] *
                                                        (1 - isequal(ty2, thread)) + # set diagonal elements to zeros
                                                        symmetric_flux_node3[tx] * shmem_split[thread, ty3] *
                                                        (1 - isequal(ty3, thread)) + # set diagonal elements to zeros
                                                        0.5f0 *
                                                        noncons_flux_node1[tx] * shmem_split[thread, ty1] +
                                                        0.5f0 *
                                                        noncons_flux_node2[tx] * shmem_split[thread, ty2] +
                                                        0.5f0 *
                                                        noncons_flux_node3[tx] * shmem_split[thread, ty3]
        end
    end

    # Synchronization is not needed here if we use only one tile
    # sync_threads()

    # Finalize the values
    for tx in axes(du, 1)
        @inbounds du[tx, ty1, ty2, ty3, k] = shmem_value[tx, ty1, ty2, ty3]
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume fluxes
function volume_flux_dgfv_kernel!(volume_flux_arr1, volume_flux_arr2, volume_flux_arr3,
                                  fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                                  u, alpha, atol, equations::AbstractEquations{3},
                                  volume_flux_dg::Any, volume_flux_fv::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^4 && k <= size(u, 5))
        u2 = size(u, 2)

        j1 = div(j - 1, u2^3) + 1
        j2 = div(rem(j - 1, u2^3), u2^2) + 1
        j3 = div(rem(j - 1, u2^2), u2) + 1
        j4 = rem(j - 1, u2) + 1

        dg_only = isapprox(alpha[k], 0, atol = atol)

        u_node = get_node_vars(u, equations, j1, j2, j3, k)
        u_node1 = get_node_vars(u, equations, j4, j2, j3, k)
        u_node2 = get_node_vars(u, equations, j1, j4, j3, k)
        u_node3 = get_node_vars(u, equations, j1, j2, j4, k)

        volume_flux_node1 = volume_flux_dg(u_node, u_node1, 1, equations)
        volume_flux_node2 = volume_flux_dg(u_node, u_node2, 2, equations)
        volume_flux_node3 = volume_flux_dg(u_node, u_node3, 3, equations)

        for ii in axes(u, 1)
            @inbounds begin
                volume_flux_arr1[ii, j1, j4, j2, j3, k] = volume_flux_node1[ii]
                volume_flux_arr2[ii, j1, j2, j4, j3, k] = volume_flux_node2[ii]
                volume_flux_arr3[ii, j1, j2, j3, j4, k] = volume_flux_node3[ii]
            end

            # Small optimization, no much performance gain
            if isequal(j1 + 1, j4) # avoid race condition
                flux_fv_node1 = volume_flux_fv(u_node, u_node1, 1, equations)

                @inbounds begin
                    fstar1_L[ii, j4, j2, j3, k] = flux_fv_node1[ii] * (1 - dg_only)
                    fstar1_R[ii, j4, j2, j3, k] = flux_fv_node1[ii] * (1 - dg_only)
                end
            end

            if isequal(j2 + 1, j4) # avoid race condition
                flux_fv_node2 = volume_flux_fv(u_node, u_node2, 2, equations)

                @inbounds begin
                    fstar2_L[ii, j1, j4, j3, k] = flux_fv_node2[ii] * (1 - dg_only)
                    fstar2_R[ii, j1, j4, j3, k] = flux_fv_node2[ii] * (1 - dg_only)
                end
            end

            if isequal(j3 + 1, j4) # avoid race condition
                flux_fv_node3 = volume_flux_fv(u_node, u_node3, 3, equations)

                @inbounds begin
                    fstar3_L[ii, j1, j2, j4, k] = flux_fv_node3[ii] * (1 - dg_only)
                    fstar3_R[ii, j1, j2, j4, k] = flux_fv_node3[ii] * (1 - dg_only)
                end
            end
        end

        # if j1 != 1 && j4 == 1 # bad
        #     u_ll = get_node_vars(u, equations, j1 - 1, j2, j3, k)
        #     u_rr = get_node_vars(u, equations, j1, j2, j3, k)
        #     flux_fv_node1 = volume_flux_fv(u_ll, u_rr, 1, equations)

        #     for ii in axes(u, 1)
        #         @inbounds begin
        #             fstar1_L[ii, j1, j2, j3, k] = flux_fv_node1[ii] * (1 - dg_only)
        #             fstar1_R[ii, j1, j2, j3, k] = flux_fv_node1[ii] * (1 - dg_only)
        #         end
        #     end
        # end

        # if j2 != 1 && j4 == 1 # bad
        #     u_ll = get_node_vars(u, equations, j1, j2 - 1, j3, k)
        #     u_rr = get_node_vars(u, equations, j1, j2, j3, k)
        #     flux_fv_node2 = volume_flux_fv(u_ll, u_rr, 2, equations)

        #     for ii in axes(u, 1)
        #         @inbounds begin
        #             fstar2_L[ii, j1, j2, j3, k] = flux_fv_node2[ii] * (1 - dg_only)
        #             fstar2_R[ii, j1, j2, j3, k] = flux_fv_node2[ii] * (1 - dg_only)
        #         end
        #     end
        # end

        # if j3 != 1 && j4 == 1 # bad
        #     u_ll = get_node_vars(u, equations, j1, j2, j3 - 1, k)
        #     u_rr = get_node_vars(u, equations, j1, j2, j3, k)
        #     flux_fv_node3 = volume_flux_fv(u_ll, u_rr, 3, equations)

        #     for ii in axes(u, 1)
        #         @inbounds begin
        #             fstar3_L[ii, j1, j2, j3, k] = flux_fv_node3[ii] * (1 - dg_only)
        #             fstar3_R[ii, j1, j2, j3, k] = flux_fv_node3[ii] * (1 - dg_only)
        #         end
        #     end
        # end
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume integrals
function volume_integral_dgfv_kernel!(du, alpha, derivative_split, inverse_weights,
                                      volume_flux_arr1, volume_flux_arr2, volume_flux_arr3,
                                      fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                                      atol, equations::AbstractEquations{3})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^3 && k <= size(du, 5))
        u2 = size(du, 2) # size(du, 2) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        @inbounds begin
            du[i, j1, j2, j3, k] = zero(eltype(du)) # initialize `du` with zeros
            alpha_element = alpha[k]
        end

        dg_only = isapprox(alpha_element, 0, atol = atol)

        for ii in axes(du, 2)
            @inbounds du[i, j1, j2, j3, k] += (derivative_split[j1, ii] *
                                               (1 - isequal(j1, ii)) * # set diagonal elements to zeros
                                               volume_flux_arr1[i, j1, ii, j2, j3, k] +
                                               derivative_split[j2, ii] *
                                               (1 - isequal(j2, ii)) * # set diagonal elements to zeros
                                               volume_flux_arr2[i, j1, j2, ii, j3, k] +
                                               derivative_split[j3, ii] *
                                               (1 - isequal(j3, ii)) * # set diagonal elements to zeros
                                               volume_flux_arr3[i, j1, j2, j3, ii, k]) * dg_only +
                                              ((1 - alpha_element) * derivative_split[j1, ii] *
                                               (1 - isequal(j1, ii)) * # set diagonal elements to zeros
                                               volume_flux_arr1[i, j1, ii, j2, j3, k] +
                                               (1 - alpha_element) * derivative_split[j2, ii] *
                                               (1 - isequal(j2, ii)) * # set diagonal elements to zeros
                                               volume_flux_arr2[i, j1, j2, ii, j3, k] +
                                               (1 - alpha_element) * derivative_split[j3, ii] *
                                               (1 - isequal(j3, ii)) * # set diagonal elements to zeros                   
                                               volume_flux_arr3[i, j1, j2, j3, ii, k]) * (1 - dg_only)
        end

        @inbounds du[i, j1, j2, j3, k] += alpha_element *
                                          (inverse_weights[j1] *
                                           (fstar1_L[i, j1 + 1, j2, j3, k] - fstar1_R[i, j1, j2, j3, k]) +
                                           inverse_weights[j2] *
                                           (fstar2_L[i, j1, j2 + 1, j3, k] - fstar2_R[i, j1, j2, j3, k]) +
                                           inverse_weights[j3] *
                                           (fstar3_L[i, j1, j2, j3 + 1, k] - fstar3_R[i, j1, j2, j3, k])) * (1 - dg_only)
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating pure DG and DG-FV volume integrals without conservative terms
function volume_flux_integral_dgfv_kernel!(du, u, alpha, atol, derivative_split, inverse_weights,
                                           equations::AbstractEquations{3},
                                           volume_flux_dg::Any, volume_flux_fv::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    # TODO: Combine `fstar` into single allocation
    shmem_split = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_fstar1 = CuDynamicSharedArray(eltype(du),
                                        (size(du, 1), tile_width + 1, tile_width, tile_width), offset)
    offset += sizeof(eltype(du)) * size(du, 1) * (tile_width + 1) * tile_width * tile_width
    shmem_fstar2 = CuDynamicSharedArray(eltype(du),
                                        (size(du, 1), tile_width, tile_width + 1, tile_width), offset)
    offset += sizeof(eltype(du)) * size(du, 1) * tile_width * (tile_width + 1) * tile_width
    shmem_fstar3 = CuDynamicSharedArray(eltype(du),
                                        (size(du, 1), tile_width, tile_width, tile_width + 1), offset)
    offset += sizeof(eltype(du)) * size(du, 1) * tile_width * tile_width * (tile_width + 1)
    shmem_value = CuDynamicSharedArray(eltype(du),
                                       (size(du, 1), tile_width, tile_width, tile_width), offset)

    # Get thread and block indices only we need save registers
    ty = threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ty1 = div(ty - 1, tile_width^2) + 1
    ty2 = div(rem(ty - 1, tile_width^2), tile_width) + 1
    ty3 = rem(rem(ty - 1, tile_width^2), tile_width) + 1

    # Load global `derivative_split` into shared memory
    # Transposed load
    @inbounds shmem_split[ty1, ty2] = derivative_split[ty2, ty1]

    # Get variables for computation
    @inbounds alpha_element = alpha[k]
    dg_only = isapprox(alpha_element, 0, atol = atol)

    # Compute FV volume fluxes
    u_node = get_node_vars(u, equations, ty1, ty2, ty3, k)
    if ty1 + 1 <= tile_width
        flux_fv_node1 = volume_flux_fv(u_node,
                                       get_node_vars(u, equations, ty1 + 1, ty2, ty3, k),
                                       1, equations)
    end
    if ty2 + 1 <= tile_width
        flux_fv_node2 = volume_flux_fv(u_node,
                                       get_node_vars(u, equations, ty1, ty2 + 1, ty3, k),
                                       2, equations)
    end
    if ty3 + 1 <= tile_width
        flux_fv_node3 = volume_flux_fv(u_node,
                                       get_node_vars(u, equations, ty1, ty2, ty3 + 1, k),
                                       3, equations)
    end

    # Initialize the values
    for tx in axes(du, 1)
        @inbounds begin
            # Initialize `du` with zeros
            shmem_value[tx, ty1, ty2, ty3] = zero(eltype(du))
            # Initialize `fstar` side columes with zeros 
            shmem_fstar1[tx, 1, ty2, ty3] = zero(eltype(du))
            shmem_fstar1[tx, tile_width + 1, ty2, ty3] = zero(eltype(du))
            shmem_fstar2[tx, ty1, 1, ty3] = zero(eltype(du))
            shmem_fstar2[tx, ty1, tile_width + 1, ty3] = zero(eltype(du))
            shmem_fstar3[tx, ty1, ty2, 1] = zero(eltype(du))
            shmem_fstar3[tx, ty1, ty2, tile_width + 1] = zero(eltype(du))
        end

        if ty1 + 1 <= tile_width
            # Set with FV volume fluxes
            @inbounds shmem_fstar1[tx, ty1 + 1, ty2, ty3] = flux_fv_node1[tx] * (1 - dg_only)
        end
        if ty2 + 1 <= tile_width
            # Set with FV volume fluxes
            @inbounds shmem_fstar2[tx, ty1, ty2 + 1, ty3] = flux_fv_node2[tx] * (1 - dg_only)
        end
        if ty3 + 1 <= tile_width
            # Set with FV volume fluxes
            @inbounds shmem_fstar3[tx, ty1, ty2, ty3 + 1] = flux_fv_node3[tx] * (1 - dg_only)
        end
    end

    sync_threads()

    # Contribute FV to the volume integrals
    for tx in axes(du, 1)
        @inbounds shmem_value[tx, ty1, ty2, ty3] += alpha_element *
                                                    (inverse_weights[ty1] *
                                                     (shmem_fstar1[tx, ty1 + 1, ty2, ty3] - shmem_fstar1[tx, ty1, ty2, ty3]) +
                                                     inverse_weights[ty2] *
                                                     (shmem_fstar2[tx, ty1, ty2 + 1, ty3] - shmem_fstar2[tx, ty1, ty2, ty3]) +
                                                     inverse_weights[ty3] *
                                                     (shmem_fstar3[tx, ty1, ty2, ty3 + 1] - shmem_fstar3[tx, ty1, ty2, ty3])) *
                                                    (1 - dg_only)
    end

    # Compute DG volume fluxes
    for thread in 1:tile_width
        volume_flux_node1 = volume_flux_dg(u_node,
                                           get_node_vars(u, equations, thread, ty2, ty3, k),
                                           1, equations)
        volume_flux_node2 = volume_flux_dg(u_node,
                                           get_node_vars(u, equations, ty1, thread, ty3, k),
                                           2, equations)
        volume_flux_node3 = volume_flux_dg(u_node,
                                           get_node_vars(u, equations, ty1, ty2, thread, k),
                                           3, equations)

        # Contribute DG to the volume integrals
        for tx in axes(du, 1)
            @inbounds shmem_value[tx, ty1, ty2, ty3] += (shmem_split[thread, ty1] *
                                                         (1 - isequal(ty1, thread)) * # set diagonal elements to zeros
                                                         volume_flux_node1[tx] +
                                                         shmem_split[thread, ty2] *
                                                         (1 - isequal(ty2, thread)) * # set diagonal elements to zeros
                                                         volume_flux_node2[tx] +
                                                         shmem_split[thread, ty3] *
                                                         (1 - isequal(ty3, thread)) * # set diagonal elements to zeros
                                                         volume_flux_node3[tx]) * dg_only +
                                                        ((1 - alpha_element) * shmem_split[thread, ty1] *
                                                         (1 - isequal(ty1, thread)) * # set diagonal elements to zeros
                                                         volume_flux_node1[tx] +
                                                         (1 - alpha_element) * shmem_split[thread, ty2] *
                                                         (1 - isequal(ty2, thread)) * # set diagonal elements to zeros
                                                         volume_flux_node2[tx] +
                                                         (1 - alpha_element) * shmem_split[thread, ty3] *
                                                         (1 - isequal(ty3, thread)) * # set diagonal elements to zeros                   
                                                         volume_flux_node3[tx]) * (1 - dg_only)
        end
    end

    # Finalize the values
    for tx in axes(du, 1)
        @inbounds du[tx, ty1, ty2, ty3, k] = shmem_value[tx, ty1, ty2, ty3]
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume fluxes
function volume_flux_dgfv_kernel!(volume_flux_arr1, volume_flux_arr2, volume_flux_arr3,
                                  noncons_flux_arr1, noncons_flux_arr2, noncons_flux_arr3,
                                  fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                                  u, alpha, atol, derivative_split,
                                  equations::AbstractEquations{3},
                                  volume_flux_dg::Any, noncons_flux_dg::Any,
                                  volume_flux_fv::Any, noncons_flux_fv::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^4 && k <= size(u, 5))
        u2 = size(u, 2)

        j1 = div(j - 1, u2^3) + 1
        j2 = div(rem(j - 1, u2^3), u2^2) + 1
        j3 = div(rem(j - 1, u2^2), u2) + 1
        j4 = rem(j - 1, u2) + 1

        dg_only = isapprox(alpha[k], 0, atol = atol)

        u_node = get_node_vars(u, equations, j1, j2, j3, k)
        u_node1 = get_node_vars(u, equations, j4, j2, j3, k)
        u_node2 = get_node_vars(u, equations, j1, j4, j3, k)
        u_node3 = get_node_vars(u, equations, j1, j2, j4, k)

        volume_flux_node1 = volume_flux_dg(u_node, u_node1, 1, equations)
        volume_flux_node2 = volume_flux_dg(u_node, u_node2, 2, equations)
        volume_flux_node3 = volume_flux_dg(u_node, u_node3, 3, equations)

        noncons_flux_node1 = noncons_flux_dg(u_node, u_node1, 1, equations)
        noncons_flux_node2 = noncons_flux_dg(u_node, u_node2, 2, equations)
        noncons_flux_node3 = noncons_flux_dg(u_node, u_node3, 3, equations)

        for ii in axes(u, 1)
            @inbounds begin
                volume_flux_arr1[ii, j1, j4, j2, j3, k] = volume_flux_node1[ii] * derivative_split[j1, j4] *
                                                          (1 - isequal(j1, j4)) # set diagonal elements to zeros
                volume_flux_arr2[ii, j1, j2, j4, j3, k] = volume_flux_node2[ii] * derivative_split[j2, j4] *
                                                          (1 - isequal(j2, j4)) # set diagonal elements to zeros
                volume_flux_arr3[ii, j1, j2, j3, j4, k] = volume_flux_node3[ii] * derivative_split[j3, j4] *
                                                          (1 - isequal(j3, j4)) # set diagonal elements to zeros

                noncons_flux_arr1[ii, j1, j4, j2, j3, k] = noncons_flux_node1[ii]
                noncons_flux_arr2[ii, j1, j2, j4, j3, k] = noncons_flux_node2[ii]
                noncons_flux_arr3[ii, j1, j2, j3, j4, k] = noncons_flux_node3[ii]
            end

            # Small optimization, no much performance gain
            if isequal(j1 + 1, j4) # avoid race condition
                f1_node = volume_flux_fv(u_node, u_node1, 1, equations)
                f1_L_node = noncons_flux_fv(u_node, u_node1, 1, equations)
                f1_R_node = noncons_flux_fv(u_node1, u_node, 1, equations)

                @inbounds begin
                    fstar1_L[ii, j4, j2, j3, k] = f1_node[ii] + 0.5f0 * f1_L_node[ii] * (1 - dg_only)
                    fstar1_R[ii, j4, j2, j3, k] = f1_node[ii] + 0.5f0 * f1_R_node[ii] * (1 - dg_only)
                end
            end

            if isequal(j2 + 1, j4) # avoid race condition
                f2_node = volume_flux_fv(u_node, u_node2, 2, equations)
                f2_L_node = noncons_flux_fv(u_node, u_node2, 2, equations)
                f2_R_node = noncons_flux_fv(u_node2, u_node, 2, equations)

                @inbounds begin
                    fstar2_L[ii, j1, j4, j3, k] = f2_node[ii] + 0.5f0 * f2_L_node[ii] * (1 - dg_only)
                    fstar2_R[ii, j1, j4, j3, k] = f2_node[ii] + 0.5f0 * f2_R_node[ii] * (1 - dg_only)
                end
            end

            if isequal(j3 + 1, j4) # avoid race condition
                f3_node = volume_flux_fv(u_node, u_node3, 3, equations)
                f3_L_node = noncons_flux_fv(u_node, u_node3, 3, equations)
                f3_R_node = noncons_flux_fv(u_node3, u_node, 3, equations)

                @inbounds begin
                    fstar3_L[ii, j1, j2, j4, k] = f3_node[ii] + 0.5f0 * f3_L_node[ii] * (1 - dg_only)
                    fstar3_R[ii, j1, j2, j4, k] = f3_node[ii] + 0.5f0 * f3_R_node[ii] * (1 - dg_only)
                end
            end
        end

        # if j1 != 1 && j4 == 1 # bad
        #     u_ll = get_node_vars(u, equations, j1 - 1, j2, j3, k)
        #     u_rr = get_node_vars(u, equations, j1, j2, j3, k)

        #     f1_node = volume_flux_fv(u_ll, u_rr, 1, equations)

        #     f1_L_node = noncons_flux_fv(u_ll, u_rr, 1, equations)
        #     f1_R_node = noncons_flux_fv(u_rr, u_ll, 1, equations)

        #     for ii in axes(u, 1)
        #         @inbounds begin
        #             fstar1_L[ii, j1, j2, j3, k] = f1_node[ii] + 0.5f0 * f1_L_node[ii] * (1 - dg_only)
        #             fstar1_R[ii, j1, j2, j3, k] = f1_node[ii] + 0.5f0 * f1_R_node[ii] * (1 - dg_only)
        #         end
        #     end
        # end

        # if j2 != 1 && j4 == 1 # bad
        #     u_ll = get_node_vars(u, equations, j1, j2 - 1, j3, k)
        #     u_rr = get_node_vars(u, equations, j1, j2, j3, k)

        #     f2_node = volume_flux_fv(u_ll, u_rr, 2, equations)

        #     f2_L_node = noncons_flux_fv(u_ll, u_rr, 2, equations)
        #     f2_R_node = noncons_flux_fv(u_rr, u_ll, 2, equations)

        #     for ii in axes(u, 1)
        #         @inbounds begin
        #             fstar2_L[ii, j1, j2, j3, k] = f2_node[ii] + 0.5f0 * f2_L_node[ii] * (1 - dg_only)
        #             fstar2_R[ii, j1, j2, j3, k] = f2_node[ii] + 0.5f0 * f2_R_node[ii] * (1 - dg_only)
        #         end
        #     end
        # end

        # if j3 != 1 && j4 == 1 # bad
        #     u_ll = get_node_vars(u, equations, j1, j2, j3 - 1, k)
        #     u_rr = get_node_vars(u, equations, j1, j2, j3, k)

        #     f3_node = volume_flux_fv(u_ll, u_rr, 3, equations)

        #     f3_L_node = noncons_flux_fv(u_ll, u_rr, 3, equations)
        #     f3_R_node = noncons_flux_fv(u_rr, u_ll, 3, equations)

        #     for ii in axes(u, 1)
        #         @inbounds begin
        #             fstar3_L[ii, j1, j2, j3, k] = f3_node[ii] + 0.5f0 * f3_L_node[ii] * (1 - dg_only)
        #             fstar3_R[ii, j1, j2, j3, k] = f3_node[ii] + 0.5f0 * f3_R_node[ii] * (1 - dg_only)
        #         end
        #     end
        # end
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume integrals
function volume_integral_dgfv_kernel!(du, alpha, derivative_split, inverse_weights,
                                      volume_flux_arr1, volume_flux_arr2, volume_flux_arr3,
                                      noncons_flux_arr1, noncons_flux_arr2, noncons_flux_arr3,
                                      fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                                      atol, equations::AbstractEquations{3})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^3 && k <= size(du, 5))
        u2 = size(du, 2) # size(du, 2) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        @inbounds begin
            du[i, j1, j2, j3, k] = zero(eltype(du)) # initialize `du` with zeros
            alpha_element = alpha[k]
        end

        dg_only = isapprox(alpha_element, 0, atol = atol)

        for ii in axes(du, 2)
            @inbounds du[i, j1, j2, j3, k] += (volume_flux_arr1[i, j1, ii, j2, j3, k] +
                                               volume_flux_arr2[i, j1, j2, ii, j3, k] +
                                               volume_flux_arr3[i, j1, j2, j3, ii, k] +
                                               0.5f0 *
                                               (derivative_split[j1, ii] * noncons_flux_arr1[i, j1, ii, j2, j3, k] +
                                                derivative_split[j2, ii] * noncons_flux_arr2[i, j1, j2, ii, j3, k] +
                                                derivative_split[j3, ii] * noncons_flux_arr3[i, j1, j2, j3, ii, k])) * dg_only +
                                              ((1 - alpha_element) *
                                               volume_flux_arr1[i, j1, ii, j2, j3, k] +
                                               (1 - alpha_element) *
                                               volume_flux_arr2[i, j1, j2, ii, j3, k] +
                                               (1 - alpha_element) *
                                               volume_flux_arr3[i, j1, j2, j3, ii, k] +
                                               0.5f0 * (1 - alpha_element) *
                                               (derivative_split[j1, ii] * noncons_flux_arr1[i, j1, ii, j2, j3, k] +
                                                derivative_split[j2, ii] * noncons_flux_arr2[i, j1, j2, ii, j3, k] +
                                                derivative_split[j3, ii] * noncons_flux_arr3[i, j1, j2, j3, ii, k])) * (1 - dg_only)
        end

        @inbounds du[i, j1, j2, j3, k] += alpha_element *
                                          (inverse_weights[j1] *
                                           (fstar1_L[i, j1 + 1, j2, j3, k] - fstar1_R[i, j1, j2, j3, k]) +
                                           inverse_weights[j2] *
                                           (fstar2_L[i, j1, j2 + 1, j3, k] - fstar2_R[i, j1, j2, j3, k]) +
                                           inverse_weights[j3] *
                                           (fstar3_L[i, j1, j2, j3 + 1, k] - fstar3_R[i, j1, j2, j3, k])) * (1 - dg_only)
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating pure DG and DG-FV volume integrals with conservative terms
function volume_flux_integral_dgfv_kernel!(du, u, alpha, atol, derivative_split, inverse_weights,
                                           equations::AbstractEquations{3},
                                           volume_flux_dg::Any, noncons_flux_dg::Any,
                                           volume_flux_fv::Any, noncons_flux_fv::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_split = CuDynamicSharedArray(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_fstar1 = CuDynamicSharedArray(eltype(du),
                                        (size(du, 1), tile_width + 1, tile_width, tile_width, 2), offset)
    offset += sizeof(eltype(du)) * size(du, 1) * (tile_width + 1) * tile_width * tile_width * 2
    shmem_fstar2 = CuDynamicSharedArray(eltype(du),
                                        (size(du, 1), tile_width, tile_width + 1, tile_width, 2), offset)
    offset += sizeof(eltype(du)) * size(du, 1) * tile_width * (tile_width + 1) * tile_width * 2
    shmem_fstar3 = CuDynamicSharedArray(eltype(du),
                                        (size(du, 1), tile_width, tile_width, tile_width + 1, 2), offset)
    offset += sizeof(eltype(du)) * size(du, 1) * tile_width * tile_width * (tile_width + 1) * 2
    shmem_value = CuDynamicSharedArray(eltype(du),
                                       (size(du, 1), tile_width, tile_width, tile_width), offset)

    # Get thread and block indices only we need save registers
    ty = threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ty1 = div(ty - 1, tile_width^2) + 1
    ty2 = div(rem(ty - 1, tile_width^2), tile_width) + 1
    ty3 = rem(rem(ty - 1, tile_width^2), tile_width) + 1

    # Load global `derivative_split` into shared memory
    # Transposed load
    @inbounds shmem_split[ty1, ty2] = derivative_split[ty2, ty1]

    # Get variables for computation
    @inbounds alpha_element = alpha[k]
    dg_only = isapprox(alpha_element, 0, atol = atol)

    # Compute FV volume fluxes
    u_node = get_node_vars(u, equations, ty1, ty2, ty3, k)
    if ty1 + 1 <= tile_width
        f1_node = volume_flux_fv(u_node,
                                 get_node_vars(u, equations, ty1 + 1, ty2, ty3, k),
                                 1, equations)
        f1_L_node = noncons_flux_fv(u_node,
                                    get_node_vars(u, equations, ty1 + 1, ty2, ty3, k),
                                    1, equations)
        f1_R_node = noncons_flux_fv(get_node_vars(u, equations, ty1 + 1, ty2, ty3, k),
                                    u_node,
                                    1, equations)
    end
    if ty2 + 1 <= tile_width
        f2_node = volume_flux_fv(u_node,
                                 get_node_vars(u, equations, ty1, ty2 + 1, ty3, k),
                                 2, equations)
        f2_L_node = noncons_flux_fv(u_node,
                                    get_node_vars(u, equations, ty1, ty2 + 1, ty3, k),
                                    2, equations)
        f2_R_node = noncons_flux_fv(get_node_vars(u, equations, ty1, ty2 + 1, ty3, k),
                                    u_node,
                                    2, equations)
    end
    if ty3 + 1 <= tile_width
        f3_node = volume_flux_fv(u_node,
                                 get_node_vars(u, equations, ty1, ty2, ty3 + 1, k),
                                 3, equations)
        f3_L_node = noncons_flux_fv(u_node,
                                    get_node_vars(u, equations, ty1, ty2, ty3 + 1, k),
                                    3, equations)
        f3_R_node = noncons_flux_fv(get_node_vars(u, equations, ty1, ty2, ty3 + 1, k),
                                    u_node,
                                    3, equations)
    end

    # Initialize the values
    for tx in axes(du, 1)
        @inbounds begin
            # Initialize `du` with zeros
            shmem_value[tx, ty1, ty2, ty3] = zero(eltype(du))

            # TODO: Remove shared memory for `fstar` and use local memory

            # Initialize `fstar` side columes with zeros (1: left)
            shmem_fstar1[tx, 1, ty2, ty3, 1] = zero(eltype(du))
            shmem_fstar1[tx, tile_width + 1, ty2, ty3, 1] = zero(eltype(du))
            shmem_fstar2[tx, ty1, 1, ty3, 1] = zero(eltype(du))
            shmem_fstar2[tx, ty1, tile_width + 1, ty3, 1] = zero(eltype(du))
            shmem_fstar3[tx, ty1, ty2, 1, 1] = zero(eltype(du))
            shmem_fstar3[tx, ty1, ty2, tile_width + 1, 1] = zero(eltype(du))

            # Initialize `fstar` side columes with zeros (2: right)
            shmem_fstar1[tx, 1, ty2, ty3, 2] = zero(eltype(du))
            shmem_fstar1[tx, tile_width + 1, ty2, ty3, 2] = zero(eltype(du))
            shmem_fstar2[tx, ty1, 1, ty3, 2] = zero(eltype(du))
            shmem_fstar2[tx, ty1, tile_width + 1, ty3, 2] = zero(eltype(du))
            shmem_fstar3[tx, ty1, ty2, 1, 2] = zero(eltype(du))
            shmem_fstar3[tx, ty1, ty2, tile_width + 1, 2] = zero(eltype(du))
        end

        if ty1 + 1 <= tile_width
            # Set with FV volume fluxes
            @inbounds begin
                shmem_fstar1[tx, ty1 + 1, ty2, ty3, 1] = f1_node[tx] + 0.5f0 * f1_L_node[tx] * (1 - dg_only)
                shmem_fstar1[tx, ty1 + 1, ty2, ty3, 2] = f1_node[tx] + 0.5f0 * f1_R_node[tx] * (1 - dg_only)
            end
        end
        if ty2 + 1 <= tile_width
            # Set with FV volume fluxes
            @inbounds begin
                shmem_fstar2[tx, ty1, ty2 + 1, ty3, 1] = f2_node[tx] + 0.5f0 * f2_L_node[tx] * (1 - dg_only)
                shmem_fstar2[tx, ty1, ty2 + 1, ty3, 2] = f2_node[tx] + 0.5f0 * f2_R_node[tx] * (1 - dg_only)
            end
        end
        if ty3 + 1 <= tile_width
            # Set with FV volume fluxes
            @inbounds begin
                shmem_fstar3[tx, ty1, ty2, ty3 + 1, 1] = f3_node[tx] + 0.5f0 * f3_L_node[tx] * (1 - dg_only)
                shmem_fstar3[tx, ty1, ty2, ty3 + 1, 2] = f3_node[tx] + 0.5f0 * f3_R_node[tx] * (1 - dg_only)
            end
        end
    end

    sync_threads()

    # Contribute FV to the volume integrals
    for tx in axes(du, 1)
        @inbounds shmem_value[tx, ty1, ty2, ty3] += alpha_element *
                                                    (inverse_weights[ty1] *
                                                     (shmem_fstar1[tx, ty1 + 1, ty2, ty3, 1] - shmem_fstar1[tx, ty1, ty2, ty3, 2]) +
                                                     inverse_weights[ty2] *
                                                     (shmem_fstar2[tx, ty1, ty2 + 1, ty3, 1] - shmem_fstar2[tx, ty1, ty2, ty3, 2]) +
                                                     inverse_weights[ty3] *
                                                     (shmem_fstar3[tx, ty1, ty2, ty3 + 1, 1] - shmem_fstar3[tx, ty1, ty2, ty3, 2])) *
                                                    (1 - dg_only)
    end

    # Compute DG volume fluxes
    for thread in 1:tile_width
        volume_flux_node1 = volume_flux_dg(u_node,
                                           get_node_vars(u, equations, thread, ty2, ty3, k),
                                           1, equations)
        volume_flux_node2 = volume_flux_dg(u_node,
                                           get_node_vars(u, equations, ty1, thread, ty3, k),
                                           2, equations)
        volume_flux_node3 = volume_flux_dg(u_node,
                                           get_node_vars(u, equations, ty1, ty2, thread, k),
                                           3, equations)

        noncons_flux_node1 = noncons_flux_dg(u_node,
                                             get_node_vars(u, equations, thread, ty2, ty3, k),
                                             1, equations)
        noncons_flux_node2 = noncons_flux_dg(u_node,
                                             get_node_vars(u, equations, ty1, thread, ty3, k),
                                             2, equations)
        noncons_flux_node3 = noncons_flux_dg(u_node,
                                             get_node_vars(u, equations, ty1, ty2, thread, k),
                                             3, equations)

        # Contribute DG to the volume integrals
        for tx in axes(du, 1)
            @inbounds shmem_value[tx, ty1, ty2, ty3] += (volume_flux_node1[tx] * shmem_split[thread, ty1] *
                                                         (1 - isequal(ty1, thread)) + # set diagonal elements to zeros
                                                         volume_flux_node2[tx] * shmem_split[thread, ty2] *
                                                         (1 - isequal(ty2, thread)) + # set diagonal elements to zeros
                                                         volume_flux_node3[tx] * shmem_split[thread, ty3] *
                                                         (1 - isequal(ty3, thread)) + # set diagonal elements to zeros
                                                         0.5f0 *
                                                         (shmem_split[thread, ty1] * noncons_flux_node1[tx] +
                                                          shmem_split[thread, ty2] * noncons_flux_node2[tx] +
                                                          shmem_split[thread, ty3] * noncons_flux_node3[tx])) * dg_only +
                                                        ((1 - alpha_element) *
                                                         volume_flux_node1[tx] * shmem_split[thread, ty1] *
                                                         (1 - isequal(ty1, thread)) + # set diagonal elements to zeros
                                                         (1 - alpha_element) *
                                                         volume_flux_node2[tx] * shmem_split[thread, ty2] *
                                                         (1 - isequal(ty2, thread)) + # set diagonal elements to zeros
                                                         (1 - alpha_element) *
                                                         volume_flux_node3[tx] * shmem_split[thread, ty3] *
                                                         (1 - isequal(ty3, thread)) + # set diagonal elements to zeros
                                                         0.5f0 * (1 - alpha_element) *
                                                         (shmem_split[thread, ty1] * noncons_flux_node1[tx] +
                                                          shmem_split[thread, ty2] * noncons_flux_node2[tx] +
                                                          shmem_split[thread, ty3] * noncons_flux_node3[tx])) * (1 - dg_only)
        end
    end

    # Finalize the values
    for tx in axes(du, 1)
        @inbounds du[tx, ty1, ty2, ty3, k] = shmem_value[tx, ty1, ty2, ty3]
    end

    return nothing
end

# Kernel for prolonging two interfaces
function prolong_interfaces_kernel!(interfaces_u, u, neighbor_ids, orientations,
                                    equations::AbstractEquations{3})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(interfaces_u, 2) * size(interfaces_u, 3)^2 && k <= size(interfaces_u, 5))
        u2 = size(u, 2) # size(interfaces_u, 3) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        @inbounds begin
            orientation = orientations[k]
            left_element = neighbor_ids[1, k]
            right_element = neighbor_ids[2, k]

            interfaces_u[1, j1, j2, j3, k] = u[j1,
                                               isequal(orientation, 1) * u2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j3,
                                               isequal(orientation, 1) * j3 + isequal(orientation, 2) * j3 + isequal(orientation, 3) * u2,
                                               left_element]
            interfaces_u[2, j1, j2, j3, k] = u[j1,
                                               isequal(orientation, 1) + isequal(orientation, 2) * j2 + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) + isequal(orientation, 3) * j3,
                                               isequal(orientation, 1) * j3 + isequal(orientation, 2) * j3 + isequal(orientation, 3),
                                               right_element]
        end
    end

    return nothing
end

# Kernel for calculating surface fluxes 
function surface_flux_kernel!(surface_flux_arr, interfaces_u, orientations,
                              equations::AbstractEquations{3}, surface_flux::Any)
    j1 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j2 = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (j1 <= size(surface_flux_arr, 2) && j2 <= size(surface_flux_arr, 3) &&
        k <= size(surface_flux_arr, 4))
        u_ll, u_rr = get_surface_node_vars(interfaces_u, equations, j1, j2, k)
        @inbounds orientation = orientations[k]

        surface_flux_node = surface_flux(u_ll, u_rr, orientation, equations)

        for ii in axes(surface_flux_arr, 1)
            @inbounds surface_flux_arr[ii, j1, j2, k] = surface_flux_node[ii]
        end
    end

    return nothing
end

# Kernel for calculating surface and both nonconservative fluxes 
function surface_noncons_flux_kernel!(surface_flux_arr, noncons_left_arr, noncons_right_arr,
                                      interfaces_u, orientations, equations::AbstractEquations{3},
                                      surface_flux::Any, nonconservative_flux::Any)
    j1 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j2 = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (j1 <= size(surface_flux_arr, 2) && j2 <= size(surface_flux_arr, 3) &&
        k <= size(surface_flux_arr, 4))
        u_ll, u_rr = get_surface_node_vars(interfaces_u, equations, j1, j2, k)
        @inbounds orientation = orientations[k]

        surface_flux_node = surface_flux(u_ll, u_rr, orientation, equations)
        noncons_left_node = nonconservative_flux(u_ll, u_rr, orientation, equations)
        noncons_right_node = nonconservative_flux(u_rr, u_ll, orientation, equations)

        for ii in axes(surface_flux_arr, 1)
            @inbounds begin
                surface_flux_arr[ii, j1, j2, k] = surface_flux_node[ii]
                noncons_left_arr[ii, j1, j2, k] = noncons_left_node[ii]
                noncons_right_arr[ii, j1, j2, k] = noncons_right_node[ii]
            end
        end
    end

    return nothing
end

# Kernel for setting interface fluxes
function interface_flux_kernel!(surface_flux_values, surface_flux_arr, neighbor_ids, orientations,
                                equations::AbstractEquations{3})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_arr, 2)^2 &&
        k <= size(surface_flux_arr, 4))
        j1 = div(j - 1, size(surface_flux_arr, 2)) + 1
        j2 = rem(j - 1, size(surface_flux_arr, 2)) + 1

        @inbounds begin
            left_id = neighbor_ids[1, k]
            right_id = neighbor_ids[2, k]

            left_direction = 2 * orientations[k]
            right_direction = 2 * orientations[k] - 1

            surface_flux_values[i, j1, j2, left_direction, left_id] = surface_flux_arr[i, j1, j2, k]
            surface_flux_values[i, j1, j2, right_direction, right_id] = surface_flux_arr[i, j1, j2, k]
        end
    end

    return nothing
end

# Kernel for setting interface fluxes
function interface_flux_kernel!(surface_flux_values, surface_flux_arr, noncons_left_arr,
                                noncons_right_arr, neighbor_ids, orientations,
                                equations::AbstractEquations{3})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_arr, 2)^2 &&
        k <= size(surface_flux_arr, 4))
        j1 = div(j - 1, size(surface_flux_arr, 2)) + 1
        j2 = rem(j - 1, size(surface_flux_arr, 2)) + 1

        @inbounds begin
            left_id = neighbor_ids[1, k]
            right_id = neighbor_ids[2, k]

            left_direction = 2 * orientations[k]
            right_direction = 2 * orientations[k] - 1

            surface_flux_values[i, j1, j2, left_direction, left_id] = surface_flux_arr[i, j1, j2, k] +
                                                                      0.5f0 *
                                                                      noncons_left_arr[i, j1, j2, k]
            surface_flux_values[i, j1, j2, right_direction, right_id] = surface_flux_arr[i, j1, j2, k] +
                                                                        0.5f0 *
                                                                        noncons_right_arr[i, j1, j2, k]
        end
    end

    return nothing
end

# Kernel for prolonging two boundaries
function prolong_boundaries_kernel!(boundaries_u, u, neighbor_ids, neighbor_sides, orientations,
                                    equations::AbstractEquations{3})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(boundaries_u, 2) * size(boundaries_u, 3)^2 && k <= size(boundaries_u, 5))
        u2 = size(u, 2) # size(boundaries_u, 3) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        @inbounds begin
            element = neighbor_ids[k]
            side = neighbor_sides[k]
            orientation = orientations[k]

            boundaries_u[1, j1, j2, j3, k] = u[j1,
                                               isequal(orientation, 1) * u2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j3,
                                               isequal(orientation, 1) * j3 + isequal(orientation, 2) * j3 + isequal(orientation, 3) * u2,
                                               element] * (2 - side) # Set to 0 instead of NaN
            boundaries_u[2, j1, j2, j3, k] = u[j1,
                                               isequal(orientation, 1) + isequal(orientation, 2) * j2 + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) + isequal(orientation, 3) * j3,
                                               isequal(orientation, 1) * j3 + isequal(orientation, 2) * j3 + isequal(orientation, 3),
                                               element] * (side - 1) # Set to 0 instead of NaN
        end
    end

    return nothing
end

# Kernel for calculating boundary fluxes
function boundary_flux_kernel!(surface_flux_values, boundaries_u, node_coordinates, t, boundary_arr,
                               indices_arr, neighbor_ids, neighbor_sides, orientations,
                               boundary_conditions::NamedTuple, equations::AbstractEquations{3},
                               surface_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(surface_flux_values, 2)^2 && k <= length(boundary_arr))
        j1 = div(j - 1, size(surface_flux_values, 2)) + 1
        j2 = rem(j - 1, size(surface_flux_values, 2)) + 1

        @inbounds begin
            boundary = boundary_arr[k]
            direction = (indices_arr[1] <= boundary) + (indices_arr[2] <= boundary) +
                        (indices_arr[3] <= boundary) + (indices_arr[4] <= boundary) +
                        (indices_arr[5] <= boundary) + (indices_arr[6] <= boundary)

            neighbor = neighbor_ids[boundary]
            side = neighbor_sides[boundary]
            orientation = orientations[boundary]
        end

        u_ll, u_rr = get_surface_node_vars(boundaries_u, equations, j1, j2, boundary)
        u_inner = (2 - side) * u_ll + (side - 1) * u_rr
        x = get_node_coords(node_coordinates, equations, j1, j2, boundary)

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
        elseif direction == 4
            boundary_flux_node = boundary_conditions[4](u_inner, orientation,
                                                        direction, x, t, surface_flux, equations)
        elseif direction == 5
            boundary_flux_node = boundary_conditions[5](u_inner, orientation,
                                                        direction, x, t, surface_flux, equations)
        else
            boundary_flux_node = boundary_conditions[6](u_inner, orientation,
                                                        direction, x, t, surface_flux, equations)
        end

        for ii in axes(surface_flux_values, 1)
            # `boundary_flux_node` can be nothing if periodic boundary condition is applied
            @inbounds surface_flux_values[ii, j1, j2, direction, neighbor] = isnothing(boundary_flux_node) ? # bad
                                                                             surface_flux_values[ii, j1,
                                                                                                 j2,
                                                                                                 direction,
                                                                                                 neighbor] :
                                                                             boundary_flux_node[ii]
        end
    end

    return nothing
end

# Kernel for copying data small to small on mortars
function prolong_mortars_small2small_kernel!(u_upper_left, u_upper_right, u_lower_left,
                                             u_lower_right, u, neighbor_ids, large_sides,
                                             orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u_upper_left, 2) && j <= size(u_upper_left, 3)^2 && k <= size(u_upper_left, 5))
        u2 = size(u, 2) # same as size(u, 2)

        j1 = div(j - 1, u2) + 1
        j2 = rem(j - 1, u2) + 1

        @inbounds begin
            large_side = large_sides[k]
            orientation = orientations[k]

            lower_left_element = neighbor_ids[1, k]
            lower_right_element = neighbor_ids[2, k]
            upper_left_element = neighbor_ids[3, k]
            upper_right_element = neighbor_ids[4, k]

            # Short index representation on large_side = 1
            idx1 = isequal(orientation, 1) + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1
            idx2 = isequal(orientation, 1) * j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2
            idx3 = isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3)

            # Short index representation on large_side = 2
            idx4 = isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1
            idx5 = isequal(orientation, 1) * j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2
            idx6 = isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2

            u_upper_left[2, i, j1, j2, k] = u[i, idx1, idx2, idx3, upper_left_element] * (2 - large_side)
            u_upper_right[2, i, j1, j2, k] = u[i, idx1, idx2, idx3, upper_right_element] * (2 - large_side)
            u_lower_left[2, i, j1, j2, k] = u[i, idx1, idx2, idx3, lower_left_element] * (2 - large_side)
            u_lower_right[2, i, j1, j2, k] = u[i, idx1, idx2, idx3, lower_right_element] * (2 - large_side)

            u_upper_left[1, i, j1, j2, k] = u[i, idx4, idx5, idx6, upper_left_element] * (large_side - 1)
            u_upper_right[1, i, j1, j2, k] = u[i, idx4, idx5, idx6, upper_right_element] * (large_side - 1)
            u_lower_left[1, i, j1, j2, k] = u[i, idx4, idx5, idx6, lower_left_element] * (large_side - 1)
            u_lower_right[1, i, j1, j2, k] = u[i, idx4, idx5, idx6, lower_right_element] * (large_side - 1)
        end
    end

    return nothing
end

# Kernel for interpolating data large to small on mortars - step 1
function prolong_mortars_large2small_kernel!(tmp_upper_left, tmp_upper_right, tmp_lower_left,
                                             tmp_lower_right, u, forward_upper, forward_lower,
                                             neighbor_ids, large_sides, orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(tmp_upper_left, 2) && j <= size(tmp_upper_left, 3)^2 && k <= size(tmp_upper_left, 5))
        u2 = size(tmp_upper_left, 3) # same as size(u, 2)

        j1 = div(j - 1, u2) + 1
        j2 = rem(j - 1, u2) + 1

        @inbounds begin
            large_side = large_sides[k]
            orientation = orientations[k]
            large_element = neighbor_ids[5, k]
        end

        leftright = large_side

        for j1j1 in axes(forward_lower, 2)
            # Short index representation on large_side = 1
            idx1 = isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1
            idx2 = isequal(orientation, 1) * j1j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2
            idx3 = isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2

            # Short index representation on large_side = 2
            idx4 = isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1
            idx5 = isequal(orientation, 1) * j1j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2
            idx6 = isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3)

            @inbounds begin
                tmp_upper_left[leftright, i, j1, j2, k] += forward_lower[j1, j1j1] *
                                                           (u[i, idx1, idx2, idx3, large_element] * (2 - large_side) +
                                                            u[i, idx4, idx5, idx6, large_element] * (large_side - 1))

                tmp_upper_right[leftright, i, j1, j2, k] += forward_upper[j1, j1j1] *
                                                            (u[i, idx1, idx2, idx3, large_element] * (2 - large_side) +
                                                             u[i, idx4, idx5, idx6, large_element] * (large_side - 1))

                tmp_lower_left[leftright, i, j1, j2, k] += forward_lower[j1, j1j1] *
                                                           (u[i, idx1, idx2, idx3, large_element] * (2 - large_side) +
                                                            u[i, idx4, idx5, idx6, large_element] * (large_side - 1))

                tmp_lower_right[leftright, i, j1, j2, k] += forward_upper[j1, j1j1] *
                                                            (u[i, idx1, idx2, idx3, large_element] * (2 - large_side) +
                                                             u[i, idx4, idx5, idx6, large_element] * (large_side - 1))
            end
        end
    end

    return nothing
end

# Kernel for interpolating data large to small on mortars - step 2
function prolong_mortars_large2small_kernel!(u_upper_left, u_upper_right, u_lower_left,
                                             u_lower_right, tmp_upper_left, tmp_upper_right,
                                             tmp_lower_left, tmp_lower_right, forward_upper,
                                             forward_lower, large_sides)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u_upper_left, 2) && j <= size(u_upper_left, 3)^2 &&
        k <= size(u_upper_left, 5))
        u2 = size(u_upper_left, 3) # size(u_upper_left, 3) == size(u, 2)

        j1 = div(j - 1, u2) + 1
        j2 = rem(j - 1, u2) + 1

        @inbounds leftright = large_sides[k]

        for j2j2 in axes(forward_upper, 2)
            @inbounds begin
                u_upper_left[leftright, i, j1, j2, k] += forward_upper[j2, j2j2] *
                                                         tmp_upper_left[leftright, i, j1, j2j2, k]

                u_upper_right[leftright, i, j1, j2, k] += forward_upper[j2, j2j2] *
                                                          tmp_upper_right[leftright, i, j1, j2j2, k]

                u_lower_left[leftright, i, j1, j2, k] += forward_lower[j2, j2j2] *
                                                         tmp_lower_left[leftright, i, j1, j2j2, k]

                u_lower_right[leftright, i, j1, j2, k] += forward_lower[j2, j2j2] *
                                                          tmp_lower_right[leftright, i, j1, j2j2, k]
            end
        end
    end

    return nothing
end

# Kernel for calculating mortar fluxes
function mortar_flux_kernel!(fstar_primary_upper_left, fstar_primary_upper_right,
                             fstar_primary_lower_left, fstar_primary_lower_right,
                             fstar_secondary_upper_left, fstar_secondary_upper_right,
                             fstar_secondary_lower_left, fstar_seondary_lower_right,
                             u_upper_left, u_upper_right, u_lower_left, u_lower_right, orientations,
                             equations::AbstractEquations{3}, surface_flux::Any)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u_upper_left, 3) && j <= size(u_upper_left, 4) && k <= length(orientations))
        u_upper_left_ll, u_upper_left_rr = get_surface_node_vars(u_upper_left, equations, i, j, k)
        u_upper_right_ll, u_upper_right_rr = get_surface_node_vars(u_upper_right, equations, i, j, k)
        u_lower_left_ll, u_lower_left_rr = get_surface_node_vars(u_lower_left, equations, i, j, k)
        u_lower_right_ll, u_lower_right_rr = get_surface_node_vars(u_lower_right, equations, i, j, k)

        @inbounds orientation = orientations[k]

        flux_upper_left_node = surface_flux(u_upper_left_ll, u_upper_left_rr, orientation,
                                            equations)
        flux_upper_right_node = surface_flux(u_upper_right_ll, u_upper_right_rr, orientation,
                                             equations)
        flux_lower_left_node = surface_flux(u_lower_left_ll, u_lower_left_rr, orientation,
                                            equations)
        flux_lower_right_node = surface_flux(u_lower_right_ll, u_lower_right_rr, orientation,
                                             equations)

        for ii in axes(fstar_primary_upper_left, 1)
            @inbounds begin
                fstar_primary_upper_left[ii, i, j, k] = flux_upper_left_node[ii]
                fstar_primary_upper_right[ii, i, j, k] = flux_upper_right_node[ii]

                fstar_primary_lower_left[ii, i, j, k] = flux_lower_left_node[ii]
                fstar_primary_lower_right[ii, i, j, k] = flux_lower_right_node[ii]

                fstar_secondary_upper_left[ii, i, j, k] = flux_upper_left_node[ii]
                fstar_secondary_upper_right[ii, i, j, k] = flux_upper_right_node[ii]

                fstar_secondary_lower_left[ii, i, j, k] = flux_lower_left_node[ii]
                fstar_seondary_lower_right[ii, i, j, k] = flux_lower_right_node[ii]
            end
        end
    end

    return nothing
end

# Kernel for calculating mortar fluxes and adding nonconservative fluxes
function mortar_flux_kernel!(fstar_primary_upper_left, fstar_primary_upper_right,
                             fstar_primary_lower_left, fstar_primary_lower_right,
                             fstar_secondary_upper_left, fstar_secondary_upper_right,
                             fstar_secondary_lower_left, fstar_seondary_lower_right,
                             u_upper_left, u_upper_right, u_lower_left, u_lower_right, orientations,
                             large_sides, equations::AbstractEquations{3}, surface_flux::Any,
                             nonconservative_flux::Any)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u_upper_left, 3) && j <= size(u_upper_left, 4) && k <= length(orientations))
        u_upper_left_ll, u_upper_left_rr = get_surface_node_vars(u_upper_left, equations, i, j, k)
        u_upper_right_ll, u_upper_right_rr = get_surface_node_vars(u_upper_right, equations, i, j, k)
        u_lower_left_ll, u_lower_left_rr = get_surface_node_vars(u_lower_left, equations, i, j, k)
        u_lower_right_ll, u_lower_right_rr = get_surface_node_vars(u_lower_right, equations, i, j, k)

        @inbounds begin
            orientation = orientations[k]
            large_side = large_sides[k]
        end

        flux_upper_left_node = surface_flux(u_upper_left_ll, u_upper_left_rr, orientation,
                                            equations)
        flux_upper_right_node = surface_flux(u_upper_right_ll, u_upper_right_rr, orientation,
                                             equations)
        flux_lower_left_node = surface_flux(u_lower_left_ll, u_lower_left_rr, orientation,
                                            equations)
        flux_lower_right_node = surface_flux(u_lower_right_ll, u_lower_right_rr, orientation,
                                             equations)

        for ii in axes(fstar_primary_upper_left, 1)
            @inbounds begin
                fstar_primary_upper_left[ii, i, j, k] = flux_upper_left_node[ii]
                fstar_primary_upper_right[ii, i, j, k] = flux_upper_right_node[ii]

                fstar_primary_lower_left[ii, i, j, k] = flux_lower_left_node[ii]
                fstar_primary_lower_right[ii, i, j, k] = flux_lower_right_node[ii]

                fstar_secondary_upper_left[ii, i, j, k] = flux_upper_left_node[ii]
                fstar_secondary_upper_right[ii, i, j, k] = flux_upper_right_node[ii]

                fstar_secondary_lower_left[ii, i, j, k] = flux_lower_left_node[ii]
                fstar_seondary_lower_right[ii, i, j, k] = flux_lower_right_node[ii]
            end
        end

        u_upper_left1 = (2 - large_side) * u_upper_left_ll + (large_side - 1) * u_upper_left_rr
        u_upper_left2 = (large_side - 1) * u_upper_left_ll + (2 - large_side) * u_upper_left_rr

        u_upper_right1 = (2 - large_side) * u_upper_right_ll + (large_side - 1) * u_upper_right_rr
        u_upper_right2 = (large_side - 1) * u_upper_right_ll + (2 - large_side) * u_upper_right_rr

        u_lower_left1 = (2 - large_side) * u_lower_left_ll + (large_side - 1) * u_lower_left_rr
        u_lower_left2 = (large_side - 1) * u_lower_left_ll + (2 - large_side) * u_lower_left_rr

        u_lower_right1 = (2 - large_side) * u_lower_right_ll + (large_side - 1) * u_lower_right_rr
        u_lower_right2 = (large_side - 1) * u_lower_right_ll + (2 - large_side) * u_lower_right_rr

        noncons_flux_primary_upper_left = nonconservative_flux(u_upper_left1, u_upper_left2,
                                                               orientation, equations)
        noncons_flux_primary_upper_right = nonconservative_flux(u_upper_right1, u_upper_right2,
                                                                orientation, equations)
        noncons_flux_primary_lower_left = nonconservative_flux(u_lower_left1, u_lower_left2,
                                                               orientation, equations)
        noncons_flux_primary_lower_right = nonconservative_flux(u_lower_right1, u_lower_right2,
                                                                orientation, equations)
        noncons_flux_secondary_upper_left = nonconservative_flux(u_upper_left2, u_upper_left1,
                                                                 orientation, equations)
        noncons_flux_secondary_upper_right = nonconservative_flux(u_upper_right2, u_upper_right1,
                                                                  orientation, equations)
        noncons_flux_secondary_lower_left = nonconservative_flux(u_lower_left2, u_lower_left1,
                                                                 orientation, equations)
        noncons_flux_secondary_lower_right = nonconservative_flux(u_lower_right2, u_lower_right1,
                                                                  orientation, equations)

        for ii in axes(fstar_primary_upper_left, 1)
            @inbounds begin
                fstar_primary_upper_left[ii, i, j, k] += 0.5f0 * noncons_flux_primary_upper_left[ii]
                fstar_primary_upper_right[ii, i, j, k] += 0.5f0 * noncons_flux_primary_upper_right[ii]

                fstar_primary_lower_left[ii, i, j, k] += 0.5f0 * noncons_flux_primary_lower_left[ii]
                fstar_primary_lower_right[ii, i, j, k] += 0.5f0 * noncons_flux_primary_lower_right[ii]

                fstar_secondary_upper_left[ii, i, j, k] += 0.5f0 * noncons_flux_secondary_upper_left[ii]
                fstar_secondary_upper_right[ii, i, j, k] += 0.5f0 * noncons_flux_secondary_upper_right[ii]

                fstar_secondary_lower_left[ii, i, j, k] += 0.5f0 * noncons_flux_secondary_lower_left[ii]
                fstar_seondary_lower_right[ii, i, j, k] += 0.5f0 * noncons_flux_secondary_lower_right[ii]
            end
        end
    end

    return nothing
end

# Kernel for copying mortar fluxes small to small and small to large - step 1
function mortar_flux_copy_to_kernel!(surface_flux_values, tmp_upper_left, tmp_upper_right,
                                     tmp_lower_left, tmp_lower_right,
                                     fstar_primary_upper_left, fstar_primary_upper_right,
                                     fstar_primary_lower_left, fstar_primary_lower_right,
                                     fstar_secondary_upper_left, fstar_secondary_upper_right,
                                     fstar_secondary_lower_left, fstar_secondary_lower_right,
                                     reverse_upper, reverse_lower, neighbor_ids, large_sides,
                                     orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_values, 2)^2 &&
        k <= length(orientations))
        j1 = div(j - 1, size(surface_flux_values, 2)) + 1
        j2 = rem(j - 1, size(surface_flux_values, 2)) + 1

        @inbounds begin
            lower_left_element = neighbor_ids[1, k]
            lower_right_element = neighbor_ids[2, k]
            upper_left_element = neighbor_ids[3, k]
            upper_right_element = neighbor_ids[4, k]
            large_element = neighbor_ids[5, k]

            large_side = large_sides[k]
            orientation = orientations[k]
        end

        # Use simple math expression to enhance the performance (against control flow), 
        # it is equivalent to, `isequal(large_side, 1) * isequal(orientation, 1) * 1 +
        #                       isequal(large_side, 1) * isequal(orientation, 2) * 3 +
        #                       isequal(large_side, 1) * isequal(orientation, 3) * 5 +
        #                       isequal(large_side, 2) * isequal(orientation, 1) * 2 +
        #                       isequal(large_side, 2) * isequal(orientation, 2) * 4 +
        #                       isequal(large_side, 2) * isequal(orientation, 3) * 6`.
        # Please also check the original code in Trixi.jl when you modify this code.
        direction = 2 * orientation + large_side - 2

        @inbounds begin
            surface_flux_values[i, j1, j2, direction, upper_left_element] = fstar_primary_upper_left[i, j1, j2, k]
            surface_flux_values[i, j1, j2, direction, upper_right_element] = fstar_primary_upper_right[i, j1, j2, k]
            surface_flux_values[i, j1, j2, direction, lower_left_element] = fstar_primary_lower_left[i, j1, j2, k]
            surface_flux_values[i, j1, j2, direction, lower_right_element] = fstar_primary_lower_right[i, j1, j2, k]
        end

        # Use simple math expression to enhance the performance (against control flow), 
        # it is equivalent to, `isequal(large_side, 1) * isequal(orientation, 1) * 2 +
        #                       isequal(large_side, 1) * isequal(orientation, 2) * 4 +
        #                       isequal(large_side, 1) * isequal(orientation, 3) * 6 +
        #                       isequal(large_side, 2) * isequal(orientation, 1) * 1 +
        #                       isequal(large_side, 2) * isequal(orientation, 2) * 3 +
        #                       isequal(large_side, 2) * isequal(orientation, 3) * 5`.
        # Please also check the original code in Trixi.jl when you modify this code.
        direction = 2 * orientation - large_side + 1

        for j1j1 in axes(reverse_upper, 2)
            @inbounds begin
                tmp_upper_left[i, j1, j2, direction, large_element] += reverse_lower[j1, j1j1] *
                                                                       fstar_secondary_upper_left[i, j1j1, j2, k]
                tmp_upper_right[i, j1, j2, direction, large_element] += reverse_upper[j1, j1j1] *
                                                                        fstar_secondary_upper_right[i, j1j1, j2, k]
                tmp_lower_left[i, j1, j2, direction, large_element] += reverse_lower[j1, j1j1] *
                                                                       fstar_secondary_lower_left[i, j1j1, j2, k]
                tmp_lower_right[i, j1, j2, direction, large_element] += reverse_upper[j1, j1j1] *
                                                                        fstar_secondary_lower_right[i, j1j1, j2, k]
            end
        end
    end

    return nothing
end

# Kernel for copying mortar fluxes small to small and small to large - step 2
function mortar_flux_copy_to_kernel!(surface_flux_values, tmp_surface_flux_values, tmp_upper_left,
                                     tmp_upper_right, tmp_lower_left, tmp_lower_right,
                                     reverse_upper, reverse_lower, neighbor_ids, large_sides,
                                     orientations, equations::AbstractEquations{3})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_values, 2)^2 &&
        k <= length(orientations))
        j1 = div(j - 1, size(surface_flux_values, 2)) + 1
        j2 = rem(j - 1, size(surface_flux_values, 2)) + 1

        @inbounds begin
            large_element = neighbor_ids[5, k]
            large_side = large_sides[k]
            orientation = orientations[k]
        end

        # See step 1 for the explanation of the following expression
        direction = 2 * orientation - large_side + 1

        for j2j2 in axes(reverse_lower, 2)
            @inbounds begin
                tmp_surface_flux_values[i, j1, j2, direction, large_element] += reverse_upper[j2, j2j2] *
                                                                                tmp_upper_left[i, j1, j2j2,
                                                                                               direction,
                                                                                               large_element]
                tmp_surface_flux_values[i, j1, j2, direction, large_element] += reverse_upper[j2, j2j2] *
                                                                                tmp_upper_right[i, j1, j2j2,
                                                                                                direction,
                                                                                                large_element]
                tmp_surface_flux_values[i, j1, j2, direction, large_element] += reverse_lower[j2, j2j2] *
                                                                                tmp_lower_left[i, j1, j2j2,
                                                                                               direction,
                                                                                               large_element]
                tmp_surface_flux_values[i, j1, j2, direction, large_element] += reverse_lower[j2, j2j2] *
                                                                                tmp_lower_right[i, j1, j2j2,
                                                                                                direction,
                                                                                                large_element]
            end
        end

        @inbounds surface_flux_values[i, j1, j2, direction, large_element] = tmp_surface_flux_values[i, j1, j2,
                                                                                                     direction,
                                                                                                     large_element]
    end

    return nothing
end

# Kernel for calculating surface integrals
function surface_integral_kernel!(du, factor_arr, surface_flux_values,
                                  equations::AbstractEquations{3})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^3 && k <= size(du, 5))
        u2 = size(du, 2) # size(du, 2) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        @inbounds begin
            du[i, j1, j2, j3, k] -= (surface_flux_values[i, j2, j3, 1, k] * isequal(j1, 1) +
                                     surface_flux_values[i, j1, j3, 3, k] * isequal(j2, 1) +
                                     surface_flux_values[i, j1, j2, 5, k] * isequal(j3, 1)) *
                                    factor_arr[1]
            du[i, j1, j2, j3, k] += (surface_flux_values[i, j2, j3, 2, k] * isequal(j1, u2) +
                                     surface_flux_values[i, j1, j3, 4, k] * isequal(j2, u2) +
                                     surface_flux_values[i, j1, j2, 6, k] * isequal(j3, u2)) *
                                    factor_arr[2]
        end
    end

    return nothing
end

# Kernel for applying inverse Jacobian 
function jacobian_kernel!(du, inverse_jacobian, equations::AbstractEquations{3})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^3 && k <= size(du, 5))
        u2 = size(du, 2) # size(du, 2) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        @inbounds du[i, j1, j2, j3, k] *= -inverse_jacobian[k]
    end

    return nothing
end

# Kernel for calculating source terms
function source_terms_kernel!(du, u, node_coordinates, t, equations::AbstractEquations{3},
                              source_terms::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(du, 2)^3 && k <= size(du, 5))
        u2 = size(u, 2) # size(du, 2) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        u_local = get_node_vars(u, equations, j1, j2, j3, k)
        x_local = get_node_coords(node_coordinates, equations, j1, j2, j3, k)

        source_terms_node = source_terms(u_local, x_local, t, equations)

        for ii in axes(du, 1)
            @inbounds du[ii, j1, j2, j3, k] += source_terms_node[ii]
        end
    end

    return nothing
end
