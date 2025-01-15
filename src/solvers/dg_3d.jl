# Everything related to a DG semidiscretization in 3D.

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

        @inbounds du[i, j1, j2, j3, k] = zero(eltype(du)) # fuse `reset_du!` here

        for ii in axes(du, 2)
            @inbounds begin
                du[i, j1, j2, j3, k] += derivative_dhat[j1, ii] * flux_arr1[i, ii, j2, j3, k]
                du[i, j1, j2, j3, k] += derivative_dhat[j2, ii] * flux_arr2[i, j1, ii, j3, k]
                du[i, j1, j2, j3, k] += derivative_dhat[j3, ii] * flux_arr3[i, j1, j2, ii, k]
            end
        end
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating fluxes and weak form
function flux_weak_form_kernel!(du, u, derivative_dhat,
                                equations::AbstractEquations{3}, flux::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_dhat = @cuDynamicSharedMem(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_flux = @cuDynamicSharedMem(eltype(du),
                                     (size(du, 1), tile_width, tile_width, tile_width, 3),
                                     offset)

    # Get thread and block indices only we need save registers
    tx, ty = threadIdx().x, threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    ty1 = div(ty - 1, tile_width^2) + 1
    ty2 = div(rem(ty - 1, tile_width^2), tile_width) + 1
    ty3 = rem(rem(ty - 1, tile_width^2), tile_width) + 1

    # Tile the computation (restrict to one tile here)
    value = zero(eltype(du))

    # Load global `derivative_dhat` into shared memory
    @inbounds shmem_dhat[ty2, ty1] = derivative_dhat[ty1, ty2] # transposed access

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
        @inbounds begin
            value += shmem_dhat[thread, ty1] * shmem_flux[tx, thread, ty2, ty3, 1] +
                     shmem_dhat[thread, ty2] * shmem_flux[tx, ty1, thread, ty3, 2] +
                     shmem_dhat[thread, ty3] * shmem_flux[tx, ty1, ty2, thread, 3]
        end
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

        @inbounds du[i, j1, j2, j3, k] = zero(eltype(du)) # fuse `reset_du!` here

        for ii in axes(du, 2)
            @inbounds begin
                du[i, j1, j2, j3, k] += derivative_split[j1, ii] *
                                        volume_flux_arr1[i, j1, ii, j2, j3, k]
                du[i, j1, j2, j3, k] += derivative_split[j2, ii] *
                                        volume_flux_arr2[i, j1, j2, ii, j3, k]
                du[i, j1, j2, j3, k] += derivative_split[j3, ii] *
                                        volume_flux_arr3[i, j1, j2, j3, ii, k]
            end
        end
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating volume fluxes and volume integrals
function volume_flux_integral_kernel!(du, u, derivative_split,
                                      equations::AbstractEquations{3}, volume_flux::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_split = @cuDynamicSharedMem(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_value = @cuDynamicSharedMem(eltype(du), (size(du, 1), tile_width, tile_width, tile_width),
                                      offset)

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
    @inbounds shmem_split[ty2, ty1] = derivative_split[ty1, ty2] # transposed access

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
            @inbounds begin
                shmem_value[tx, ty1, ty2, ty3] += shmem_split[thread, ty1] * volume_flux_node1[tx] +
                                                  shmem_split[thread, ty2] * volume_flux_node2[tx] +
                                                  shmem_split[thread, ty3] * volume_flux_node3[tx]
            end
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
                symmetric_flux_arr1[ii, j1, j4, j2, j3, k] = derivative_split[j1, j4] *
                                                             symmetric_flux_node1[ii]
                symmetric_flux_arr2[ii, j1, j2, j4, j3, k] = derivative_split[j2, j4] *
                                                             symmetric_flux_node2[ii]
                symmetric_flux_arr3[ii, j1, j2, j3, j4, k] = derivative_split[j3, j4] *
                                                             symmetric_flux_node3[ii]

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

        @inbounds du[i, j1, j2, j3, k] = zero(eltype(du)) # fuse `reset_du!` here
        integral_contribution = zero(eltype(du))

        for ii in axes(du, 2)
            @inbounds begin
                du[i, j1, j2, j3, k] += symmetric_flux_arr1[i, j1, ii, j2, j3, k]
                du[i, j1, j2, j3, k] += symmetric_flux_arr2[i, j1, j2, ii, j3, k]
                du[i, j1, j2, j3, k] += symmetric_flux_arr3[i, j1, j2, j3, ii, k]

                integral_contribution += derivative_split[j1, ii] *
                                         noncons_flux_arr1[i, j1, ii, j2, j3, k]
                integral_contribution += derivative_split[j2, ii] *
                                         noncons_flux_arr2[i, j1, j2, ii, j3, k]
                integral_contribution += derivative_split[j3, ii] *
                                         noncons_flux_arr3[i, j1, j2, j3, ii, k]
            end
        end

        @inbounds du[i, j1, j2, j3, k] += 0.5f0 * integral_contribution
    end

    return nothing
end

############################################################################## New optimization
# Kernel for calculating symmetric and nonconservative volume fluxes and 
# corresponding volume integrals
function noncons_volume_flux_integral_kernel!(du, u, derivative_split, derivative_split_zero,
                                              equations::AbstractEquations{3},
                                              symmetric_flux::Any, nonconservative_flux::Any)
    # Set tile width
    tile_width = size(du, 2)
    offset = 0 # offset bytes for shared memory

    # Allocate dynamic shared memory
    shmem_split = @cuDynamicSharedMem(eltype(du), (tile_width, tile_width))
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_szero = @cuDynamicSharedMem(eltype(du), (tile_width, tile_width),
                                      offset)
    offset += sizeof(eltype(du)) * tile_width^2
    shmem_value = @cuDynamicSharedMem(eltype(du), (size(du, 1), tile_width, tile_width, tile_width),
                                      offset)

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
    # Transposed access
    @inbounds begin
        shmem_split[ty2, ty1] = derivative_split[ty1, ty2]
        shmem_szero[ty2, ty1] = derivative_split_zero[ty1, ty2]
    end

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
            @inbounds begin
                shmem_value[tx, ty1, ty2, ty3] += symmetric_flux_node1[tx] * shmem_szero[thread, ty1] +
                                                  symmetric_flux_node2[tx] * shmem_szero[thread, ty2] +
                                                  symmetric_flux_node3[tx] * shmem_szero[thread, ty3] +
                                                  0.5f0 * noncons_flux_node1[tx] * shmem_split[thread, ty1] +
                                                  0.5f0 * noncons_flux_node2[tx] * shmem_split[thread, ty2] +
                                                  0.5f0 * noncons_flux_node3[tx] * shmem_split[thread, ty3]
            end
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
                                  u, element_ids_dgfv, equations::AbstractEquations{3},
                                  volume_flux_dg::Any, volume_flux_fv::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^4 && k <= size(u, 5))
        # length(element_ids_dgfv) == size(u, 5)
        u2 = size(u, 2)

        j1 = div(j - 1, u2^3) + 1
        j2 = div(rem(j - 1, u2^3), u2^2) + 1
        j3 = div(rem(j - 1, u2^2), u2) + 1
        j4 = rem(j - 1, u2) + 1

        @inbounds element_dgfv = element_ids_dgfv[k] # check if `element_dgfv` is zero

        # The sets of `get_node_vars` operations may be combined
        # into a single set of operation for better performance (to be explored).

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
        end

        if j1 != 1 && j4 == 1 && element_dgfv != 0 # bad
            u_ll = get_node_vars(u, equations, j1 - 1, j2, j3, element_dgfv)
            u_rr = get_node_vars(u, equations, j1, j2, j3, element_dgfv)
            flux_fv_node1 = volume_flux_fv(u_ll, u_rr, 1, equations)

            for ii in axes(u, 1)
                @inbounds begin
                    fstar1_L[ii, j1, j2, j3, element_dgfv] = flux_fv_node1[ii]
                    fstar1_R[ii, j1, j2, j3, element_dgfv] = flux_fv_node1[ii]
                end
            end
        end

        if j2 != 1 && j4 == 1 && element_dgfv != 0 # bad
            u_ll = get_node_vars(u, equations, j1, j2 - 1, j3, element_dgfv)
            u_rr = get_node_vars(u, equations, j1, j2, j3, element_dgfv)
            flux_fv_node2 = volume_flux_fv(u_ll, u_rr, 2, equations)

            for ii in axes(u, 1)
                @inbounds begin
                    fstar2_L[ii, j1, j2, j3, element_dgfv] = flux_fv_node2[ii]
                    fstar2_R[ii, j1, j2, j3, element_dgfv] = flux_fv_node2[ii]
                end
            end
        end

        if j3 != 1 && j4 == 1 && element_dgfv != 0 # bad
            u_ll = get_node_vars(u, equations, j1, j2, j3 - 1, element_dgfv)
            u_rr = get_node_vars(u, equations, j1, j2, j3, element_dgfv)
            flux_fv_node3 = volume_flux_fv(u_ll, u_rr, 3, equations)

            for ii in axes(u, 1)
                @inbounds begin
                    fstar3_L[ii, j1, j2, j3, element_dgfv] = flux_fv_node3[ii]
                    fstar3_R[ii, j1, j2, j3, element_dgfv] = flux_fv_node3[ii]
                end
            end
        end
    end

    return nothing
end

# Kernel for calculating DG volume integral contribution
function volume_integral_dg_kernel!(du, element_ids_dg, element_ids_dgfv, alpha, derivative_split,
                                    volume_flux_arr1, volume_flux_arr2, volume_flux_arr3,
                                    equations::AbstractEquations{3})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^3 && k <= size(du, 5))
        # length(element_ids_dg) == size(u, 5)
        u2 = size(du, 2) # size(du, 2) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        @inbounds du[i, j1, j2, j3, k] = zero(eltype(du)) # fuse `reset_du!` here

        @inbounds begin
            element_dg = element_ids_dg[k] # check if `element_dg` is zero
            element_dgfv = element_ids_dgfv[k] # check if `element_dgfv` is zero
            alpha_element = alpha[k]
        end

        if element_dg != 0 # bad
            for ii in axes(du, 2)
                @inbounds begin
                    du[i, j1, j2, j3, element_dg] += derivative_split[j1, ii] *
                                                     volume_flux_arr1[i, j1, ii, j2, j3, element_dg]
                    du[i, j1, j2, j3, element_dg] += derivative_split[j2, ii] *
                                                     volume_flux_arr2[i, j1, j2, ii, j3, element_dg]
                    du[i, j1, j2, j3, element_dg] += derivative_split[j3, ii] *
                                                     volume_flux_arr3[i, j1, j2, j3, ii, element_dg]
                end
            end
        end

        if element_dgfv != 0 # bad
            for ii in axes(du, 2)
                @inbounds begin
                    du[i, j1, j2, j3, element_dgfv] += (1 - alpha_element) *
                                                       derivative_split[j1, ii] *
                                                       volume_flux_arr1[i, j1, ii, j2, j3,
                                                                        element_dgfv]
                    du[i, j1, j2, j3, element_dgfv] += (1 - alpha_element) *
                                                       derivative_split[j2, ii] *
                                                       volume_flux_arr2[i, j1, j2, ii, j3,
                                                                        element_dgfv]
                    du[i, j1, j2, j3, element_dgfv] += (1 - alpha_element) *
                                                       derivative_split[j3, ii] *
                                                       volume_flux_arr3[i, j1, j2, j3, ii,
                                                                        element_dgfv]
                end
            end
        end
    end

    return nothing
end

# Kernel for calculating pure DG and DG-FV volume fluxes
function volume_flux_dgfv_kernel!(volume_flux_arr1, volume_flux_arr2, volume_flux_arr3,
                                  noncons_flux_arr1, noncons_flux_arr2, noncons_flux_arr3,
                                  fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                                  u, element_ids_dgfv, derivative_split,
                                  equations::AbstractEquations{3},
                                  volume_flux_dg::Any, nonconservative_flux_dg::Any,
                                  volume_flux_fv::Any, nonconservative_flux_fv::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^4 && k <= size(u, 5))
        # length(element_ids_dgfv) == size(u, 5)
        u2 = size(u, 2)

        j1 = div(j - 1, u2^3) + 1
        j2 = div(rem(j - 1, u2^3), u2^2) + 1
        j3 = div(rem(j - 1, u2^2), u2) + 1
        j4 = rem(j - 1, u2) + 1

        @inbounds element_dgfv = element_ids_dgfv[k] # check if `element_dgfv` is zero

        # The sets of `get_node_vars` operations may be combined
        # into a single set of operation for better performance (to be explored).

        u_node = get_node_vars(u, equations, j1, j2, j3, k)
        u_node1 = get_node_vars(u, equations, j4, j2, j3, k)
        u_node2 = get_node_vars(u, equations, j1, j4, j3, k)
        u_node3 = get_node_vars(u, equations, j1, j2, j4, k)

        volume_flux_node1 = volume_flux_dg(u_node, u_node1, 1, equations)
        volume_flux_node2 = volume_flux_dg(u_node, u_node2, 2, equations)
        volume_flux_node3 = volume_flux_dg(u_node, u_node3, 3, equations)

        noncons_flux_node1 = nonconservative_flux_dg(u_node, u_node1, 1, equations)
        noncons_flux_node2 = nonconservative_flux_dg(u_node, u_node2, 2, equations)
        noncons_flux_node3 = nonconservative_flux_dg(u_node, u_node3, 3, equations)

        for ii in axes(u, 1)
            @inbounds begin
                volume_flux_arr1[ii, j1, j4, j2, j3, k] = derivative_split[j1, j4] *
                                                          volume_flux_node1[ii]
                volume_flux_arr2[ii, j1, j2, j4, j3, k] = derivative_split[j2, j4] *
                                                          volume_flux_node2[ii]
                volume_flux_arr3[ii, j1, j2, j3, j4, k] = derivative_split[j3, j4] *
                                                          volume_flux_node3[ii]

                noncons_flux_arr1[ii, j1, j4, j2, j3, k] = noncons_flux_node1[ii]
                noncons_flux_arr2[ii, j1, j2, j4, j3, k] = noncons_flux_node2[ii]
                noncons_flux_arr3[ii, j1, j2, j3, j4, k] = noncons_flux_node3[ii]
            end
        end

        if j1 != 1 && j4 == 1 && element_dgfv != 0 # bad
            u_ll = get_node_vars(u, equations, j1 - 1, j2, j3, element_dgfv)
            u_rr = get_node_vars(u, equations, j1, j2, j3, element_dgfv)

            f1_node = volume_flux_fv(u_ll, u_rr, 1, equations)

            f1_L_node = nonconservative_flux_fv(u_ll, u_rr, 1, equations)
            f1_R_node = nonconservative_flux_fv(u_rr, u_ll, 1, equations)

            for ii in axes(u, 1)
                @inbounds begin
                    fstar1_L[ii, j1, j2, j3, element_dgfv] = f1_node[ii] + 0.5f0 * f1_L_node[ii]
                    fstar1_R[ii, j1, j2, j3, element_dgfv] = f1_node[ii] + 0.5f0 * f1_R_node[ii]
                end
            end
        end

        if j2 != 1 && j4 == 1 && element_dgfv != 0 # bad
            u_ll = get_node_vars(u, equations, j1, j2 - 1, j3, element_dgfv)
            u_rr = get_node_vars(u, equations, j1, j2, j3, element_dgfv)

            f2_node = volume_flux_fv(u_ll, u_rr, 2, equations)

            f2_L_node = nonconservative_flux_fv(u_ll, u_rr, 2, equations)
            f2_R_node = nonconservative_flux_fv(u_rr, u_ll, 2, equations)

            for ii in axes(u, 1)
                @inbounds begin
                    fstar2_L[ii, j1, j2, j3, element_dgfv] = f2_node[ii] + 0.5f0 * f2_L_node[ii]
                    fstar2_R[ii, j1, j2, j3, element_dgfv] = f2_node[ii] + 0.5f0 * f2_R_node[ii]
                end
            end
        end

        if j3 != 1 && j4 == 1 && element_dgfv != 0 # bad
            u_ll = get_node_vars(u, equations, j1, j2, j3 - 1, element_dgfv)
            u_rr = get_node_vars(u, equations, j1, j2, j3, element_dgfv)

            f3_node = volume_flux_fv(u_ll, u_rr, 3, equations)

            f3_L_node = nonconservative_flux_fv(u_ll, u_rr, 3, equations)
            f3_R_node = nonconservative_flux_fv(u_rr, u_ll, 3, equations)

            for ii in axes(u, 1)
                @inbounds begin
                    fstar3_L[ii, j1, j2, j3, element_dgfv] = f3_node[ii] + 0.5f0 * f3_L_node[ii]
                    fstar3_R[ii, j1, j2, j3, element_dgfv] = f3_node[ii] + 0.5f0 * f3_R_node[ii]
                end
            end
        end
    end

    return nothing
end

# Kernel for calculating DG volume integral contribution
function volume_integral_dg_kernel!(du, element_ids_dg, element_ids_dgfv, alpha, derivative_split,
                                    volume_flux_arr1, volume_flux_arr2, volume_flux_arr3,
                                    noncons_flux_arr1, noncons_flux_arr2, noncons_flux_arr3,
                                    equations::AbstractEquations{3})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^3 && k <= size(du, 5))
        # length(element_ids_dg) == size(u, 5)
        u2 = size(du, 2) # size(du, 2) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        @inbounds du[i, j1, j2, j3, k] = zero(eltype(du)) # fuse `reset_du!` here

        @inbounds begin
            element_dg = element_ids_dg[k] # check if `element_dg` is zero
            element_dgfv = element_ids_dgfv[k] # check if `element_dgfv` is zero
            alpha_element = alpha[k]
        end

        if element_dg != 0 # bad
            integral_contribution = zero(eltype(du))

            for ii in axes(du, 2)
                @inbounds begin
                    du[i, j1, j2, j3, element_dg] += volume_flux_arr1[i, j1, ii, j2, j3, element_dg]
                    du[i, j1, j2, j3, element_dg] += volume_flux_arr2[i, j1, j2, ii, j3, element_dg]
                    du[i, j1, j2, j3, element_dg] += volume_flux_arr3[i, j1, j2, j3, ii, element_dg]

                    integral_contribution += derivative_split[j1, ii] *
                                             noncons_flux_arr1[i, j1, ii, j2, j3,
                                                               element_dg]
                    integral_contribution += derivative_split[j2, ii] *
                                             noncons_flux_arr2[i, j1, j2, ii, j3,
                                                               element_dg]
                    integral_contribution += derivative_split[j3, ii] *
                                             noncons_flux_arr3[i, j1, j2, j3, ii,
                                                               element_dg]
                end
            end

            @inbounds du[i, j1, j2, j3, element_dg] += 0.5f0 * integral_contribution
        end

        if element_dgfv != 0 # bad
            integral_contribution = zero(eltype(du))

            for ii in axes(du, 2)
                @inbounds begin
                    du[i, j1, j2, j3, element_dgfv] += (1 - alpha_element) *
                                                       volume_flux_arr1[i, j1, ii, j2, j3,
                                                                        element_dgfv]
                    du[i, j1, j2, j3, element_dgfv] += (1 - alpha_element) *
                                                       volume_flux_arr2[i, j1, j2, ii, j3,
                                                                        element_dgfv]
                    du[i, j1, j2, j3, element_dgfv] += (1 - alpha_element) *
                                                       volume_flux_arr3[i, j1, j2, j3, ii,
                                                                        element_dgfv]

                    integral_contribution += derivative_split[j1, ii] *
                                             noncons_flux_arr1[i, j1, ii, j2, j3,
                                                               element_dgfv]
                    integral_contribution += derivative_split[j2, ii] *
                                             noncons_flux_arr2[i, j1, j2, ii, j3,
                                                               element_dgfv]
                    integral_contribution += derivative_split[j3, ii] *
                                             noncons_flux_arr3[i, j1, j2, j3, ii,
                                                               element_dgfv]
                end
            end

            @inbounds du[i, j1, j2, j3, element_dgfv] += 0.5f0 * (1 - alpha_element) * integral_contribution
        end
    end

    return nothing
end

# Kernel for calculating FV volume integral contribution 
function volume_integral_fv_kernel!(du, fstar1_L, fstar1_R, fstar2_L, fstar2_R,
                                    fstar3_L, fstar3_R, inverse_weights, element_ids_dgfv, alpha)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(du, 2)^3 && k <= size(du, 5))
        # length(element_ids_dgfv) == size(du, 5)
        u2 = size(du, 2) # size(du, 2) == size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        @inbounds begin
            element_dgfv = element_ids_dgfv[k] # check if `element_dgfv` is zero
            alpha_element = alpha[k]
        end

        if element_dgfv != 0 # bad
            for ii in axes(du, 1)
                @inbounds du[ii, j1, j2, j3, element_dgfv] += (alpha_element *
                                                               (inverse_weights[j1] *
                                                                (fstar1_L[ii, j1 + 1, j2, j3,
                                                                          element_dgfv] -
                                                                 fstar1_R[ii, j1, j2, j3, element_dgfv]) +
                                                                inverse_weights[j2] *
                                                                (fstar2_L[ii, j1, j2 + 1, j3,
                                                                          element_dgfv] -
                                                                 fstar2_R[ii, j1, j2, j3, element_dgfv]) +
                                                                inverse_weights[j3] *
                                                                (fstar3_L[ii, j1, j2, j3 + 1,
                                                                          element_dgfv] -
                                                                 fstar3_R[ii, j1, j2, j3, element_dgfv])))
            end
        end
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
        u2 = size(u, 2) # size(u_upper_left, 3) == size(u, 2)

        j1 = div(j - 1, u2) + 1
        j2 = rem(j - 1, u2) + 1

        @inbounds begin
            large_side = large_sides[k]
            orientation = orientations[k]

            lower_left_element = neighbor_ids[1, k]
            lower_right_element = neighbor_ids[2, k]
            upper_left_element = neighbor_ids[3, k]
            upper_right_element = neighbor_ids[4, k]

            u_upper_left[2, i, j1, j2, k] = u[i,
                                              isequal(orientation, 1) + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1,
                                              isequal(orientation, 1) * j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2,
                                              isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3),
                                              upper_left_element] * (2 - large_side)

            u_upper_right[2, i, j1, j2, k] = u[i,
                                               isequal(orientation, 1) + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1,
                                               isequal(orientation, 1) * j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3),
                                               upper_right_element] * (2 - large_side)

            u_lower_left[2, i, j1, j2, k] = u[i,
                                              isequal(orientation, 1) + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1,
                                              isequal(orientation, 1) * j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2,
                                              isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3),
                                              lower_left_element] * (2 - large_side)

            u_lower_right[2, i, j1, j2, k] = u[i,
                                               isequal(orientation, 1) + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1,
                                               isequal(orientation, 1) * j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3),
                                               lower_right_element] * (2 - large_side)

            u_upper_left[1, i, j1, j2, k] = u[i,
                                              isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1,
                                              isequal(orientation, 1) * j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2,
                                              isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2,
                                              upper_left_element] * (large_side - 1)

            u_upper_right[1, i, j1, j2, k] = u[i,
                                               isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1,
                                               isequal(orientation, 1) * j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2,
                                               upper_right_element] * (large_side - 1)

            u_lower_left[1, i, j1, j2, k] = u[i,
                                              isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1,
                                              isequal(orientation, 1) * j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2,
                                              isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2,
                                              lower_left_element] * (large_side - 1)

            u_lower_right[1, i, j1, j2, k] = u[i,
                                               isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1,
                                               isequal(orientation, 1) * j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2,
                                               lower_right_element] * (large_side - 1)
        end
    end

    return nothing
end

# # Kernel for interpolating data large to small on mortars - step 1
# function prolong_mortars_large2small_kernel!(tmp_upper_left, tmp_upper_right, tmp_lower_left,
#                                              tmp_lower_right, u, forward_upper,
#                                              forward_lower, neighbor_ids, large_sides, orientations)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
#     k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

#     if (i <= size(tmp_upper_left, 2) && j <= size(tmp_upper_left, 3)^2 &&
#         k <= size(tmp_upper_left, 5))
#         u2 = size(tmp_upper_left, 3) # size(tmp_upper_left, 3) == size(u, 2)

#         j1 = div(j - 1, u2) + 1
#         j2 = rem(j - 1, u2) + 1

#         large_side = large_sides[k]
#         orientation = orientations[k]
#         large_element = neighbor_ids[5, k]

#         leftright = large_side

#         @inbounds begin
#             for j1j1 in axes(forward_lower, 2)
#                 tmp_upper_left[leftright, i, j1, j2, k] += forward_lower[j1, j1j1] *
#                                                            u[i,
#                                                              isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
#                                                              isequal(orientation, 1) * j1j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2,
#                                                              isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2,
#                                                              large_element] * (2 - large_side)

#                 tmp_upper_right[leftright, i, j1, j2, k] += forward_upper[j1, j1j1] *
#                                                             u[i,
#                                                               isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
#                                                               isequal(orientation, 1) * j1j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2,
#                                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2,
#                                                               large_element] * (2 - large_side)

#                 tmp_lower_left[leftright, i, j1, j2, k] += forward_lower[j1, j1j1] *
#                                                            u[i,
#                                                              isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
#                                                              isequal(orientation, 1) * j1j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2,
#                                                              isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2,
#                                                              large_element] * (2 - large_side)

#                 tmp_lower_right[leftright, i, j1, j2, k] += forward_upper[j1, j1j1] *
#                                                             u[i,
#                                                               isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
#                                                               isequal(orientation, 1) * j1j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2,
#                                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2,
#                                                               large_element] * (2 - large_side)
#             end

#             for j1j1 in axes(forward_lower, 2)
#                 tmp_upper_left[leftright, i, j1, j2, k] += forward_lower[j1, j1j1] *
#                                                            u[i,
#                                                              isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
#                                                              isequal(orientation, 1) * j1j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2,
#                                                              isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3),
#                                                              large_element] * (large_side - 1)

#                 tmp_upper_right[leftright, i, j1, j2, k] += forward_upper[j1, j1j1] *
#                                                             u[i,
#                                                               isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
#                                                               isequal(orientation, 1) * j1j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2,
#                                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3),
#                                                               large_element] * (large_side - 1)

#                 tmp_lower_left[leftright, i, j1, j2, k] += forward_lower[j1, j1j1] *
#                                                            u[i,
#                                                              isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
#                                                              isequal(orientation, 1) * j1j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2,
#                                                              isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3),
#                                                              large_element] * (large_side - 1)

#                 tmp_lower_right[leftright, i, j1, j2, k] += forward_upper[j1, j1j1] *
#                                                             u[i,
#                                                               isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
#                                                               isequal(orientation, 1) * j1j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2,
#                                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3),
#                                                               large_element] * (large_side - 1)
#             end
#         end
#     end

#     return nothing
# end

# # Kernel for interpolating data large to small on mortars - step 2
# function prolong_mortars_large2small_kernel!(u_upper_left, u_upper_right, u_lower_left,
#                                              u_lower_right, tmp_upper_left, tmp_upper_right,
#                                              tmp_lower_left, tmp_lower_right, forward_upper,
#                                              forward_lower, large_sides)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
#     k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

#     if (i <= size(u_upper_left, 2) && j <= size(u_upper_left, 3)^2 &&
#         k <= size(u_upper_left, 5))
#         u2 = size(u_upper_left, 3) # size(u_upper_left, 3) == size(u, 2)

#         j1 = div(j - 1, u2) + 1
#         j2 = rem(j - 1, u2) + 1

#         leftright = large_sides[k]

#         @inbounds begin
#             for j2j2 in axes(forward_upper, 2)
#                 u_upper_left[leftright, i, j1, j2, k] += forward_upper[j2, j2j2] *
#                                                          tmp_upper_left[leftright, i, j1, j2j2, k]

#                 u_upper_right[leftright, i, j1, j2, k] += forward_upper[j2, j2j2] *
#                                                           tmp_upper_right[leftright, i, j1, j2j2, k]

#                 u_lower_left[leftright, i, j1, j2, k] += forward_lower[j2, j2j2] *
#                                                          tmp_lower_left[leftright, i, j1, j2j2, k]

#                 u_lower_right[leftright, i, j1, j2, k] += forward_lower[j2, j2j2] *
#                                                           tmp_lower_right[leftright, i, j1, j2j2, k]
#             end
#         end
#     end

#     return nothing
# end

# Kernel for interpolating data large to small on mortars (optimized)
function prolong_mortars_large2small_kernel!(u_upper_left, u_upper_right, u_lower_left, u_lower_right,
                                             tmp_upper_left, tmp_upper_right, tmp_lower_left, tmp_lower_right,
                                             u, forward_upper, forward_lower, neighbor_ids, large_sides,
                                             orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    # Loop stride for each dimension
    stride_x = gridDim().x * blockDim().x
    stride_y = gridDim().y * blockDim().y
    stride_z = gridDim().z * blockDim().z

    # Cooperative kernel needs stride loops to handle the constrained launch size
    while i <= size(tmp_upper_left, 2)
        while j <= size(tmp_upper_left, 3)^2
            while k <= size(tmp_upper_left, 5)
                u2 = size(tmp_upper_left, 3) # size(tmp_upper_left, 3) == size(u, 2)

                j1 = div(j - 1, u2) + 1
                j2 = rem(j - 1, u2) + 1

                @inbounds begin
                    large_side = large_sides[k]
                    orientation = orientations[k]
                    large_element = neighbor_ids[5, k]
                end

                leftright = large_side

                for j1j1 in axes(forward_lower, 2)
                    @inbounds begin
                        tmp_upper_left[leftright, i, j1, j2, k] += forward_lower[j1, j1j1] *
                                                                   u[i,
                                                                     isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
                                                                     isequal(orientation, 1) * j1j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2,
                                                                     isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2,
                                                                     large_element] * (2 - large_side)

                        tmp_upper_right[leftright, i, j1, j2, k] += forward_upper[j1, j1j1] *
                                                                    u[i,
                                                                      isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
                                                                      isequal(orientation, 1) * j1j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2,
                                                                      isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2,
                                                                      large_element] * (2 - large_side)

                        tmp_lower_left[leftright, i, j1, j2, k] += forward_lower[j1, j1j1] *
                                                                   u[i,
                                                                     isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
                                                                     isequal(orientation, 1) * j1j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2,
                                                                     isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2,
                                                                     large_element] * (2 - large_side)

                        tmp_lower_right[leftright, i, j1, j2, k] += forward_upper[j1, j1j1] *
                                                                    u[i,
                                                                      isequal(orientation, 1) * u2 + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
                                                                      isequal(orientation, 1) * j1j1 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j2,
                                                                      isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * u2,
                                                                      large_element] * (2 - large_side)
                    end
                end

                for j1j1 in axes(forward_lower, 2)
                    @inbounds begin
                        tmp_upper_left[leftright, i, j1, j2, k] += forward_lower[j1, j1j1] *
                                                                   u[i,
                                                                     isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
                                                                     isequal(orientation, 1) * j1j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2,
                                                                     isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation,
                                                                                                                                           3),
                                                                     large_element] * (large_side - 1)

                        tmp_upper_right[leftright, i, j1, j2, k] += forward_upper[j1, j1j1] *
                                                                    u[i,
                                                                      isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
                                                                      isequal(orientation, 1) * j1j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2,
                                                                      isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation,
                                                                                                                                            3),
                                                                      large_element] * (large_side - 1)

                        tmp_lower_left[leftright, i, j1, j2, k] += forward_lower[j1, j1j1] *
                                                                   u[i,
                                                                     isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
                                                                     isequal(orientation, 1) * j1j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2,
                                                                     isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation,
                                                                                                                                           3),
                                                                     large_element] * (large_side - 1)

                        tmp_lower_right[leftright, i, j1, j2, k] += forward_upper[j1, j1j1] *
                                                                    u[i,
                                                                      isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
                                                                      isequal(orientation, 1) * j1j1 + isequal(orientation, 2) + isequal(orientation, 3) * j2,
                                                                      isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation,
                                                                                                                                            3),
                                                                      large_element] * (large_side - 1)
                    end
                end

                # Grid scope synchronization
                grid = CG.this_grid()
                CG.sync(grid)

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
                k += stride_z
            end
            j += stride_y
        end
        i += stride_x
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

# # Kernel for copying mortar fluxes small to small and small to large - step 1
# function mortar_flux_copy_to_kernel!(surface_flux_values, tmp_upper_left, tmp_upper_right,
#                                      tmp_lower_left, tmp_lower_right,
#                                      fstar_primary_upper_left, fstar_primary_upper_right,
#                                      fstar_primary_lower_left, fstar_primary_lower_right,
#                                      fstar_secondary_upper_left, fstar_secondary_upper_right,
#                                      fstar_secondary_lower_left, fstar_secondary_lower_right,
#                                      reverse_upper, reverse_lower, neighbor_ids, large_sides,
#                                      orientations)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
#     k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

#     if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_values, 2)^2 &&
#         k <= length(orientations))
#         j1 = div(j - 1, size(surface_flux_values, 2)) + 1
#         j2 = rem(j - 1, size(surface_flux_values, 2)) + 1

#         lower_left_element = neighbor_ids[1, k]
#         lower_right_element = neighbor_ids[2, k]
#         upper_left_element = neighbor_ids[3, k]
#         upper_right_element = neighbor_ids[4, k]
#         large_element = neighbor_ids[5, k]

#         large_side = large_sides[k]
#         orientation = orientations[k]

#         # Use simple math expression to enhance the performance (against control flow), 
#         # it is equivalent to, `isequal(large_side, 1) * isequal(orientation, 1) * 1 +
#         #                       isequal(large_side, 1) * isequal(orientation, 2) * 3 +
#         #                       isequal(large_side, 1) * isequal(orientation, 3) * 5 +
#         #                       isequal(large_side, 2) * isequal(orientation, 1) * 2 +
#         #                       isequal(large_side, 2) * isequal(orientation, 2) * 4 +
#         #                       isequal(large_side, 2) * isequal(orientation, 3) * 6`.
#         # Please also check the original code in Trixi.jl when you modify this code.
#         direction = 2 * orientation + large_side - 2

#         surface_flux_values[i, j1, j2, direction, upper_left_element] = fstar_primary_upper_left[i, j1, j2, k]
#         surface_flux_values[i, j1, j2, direction, upper_right_element] = fstar_primary_upper_right[i, j1, j2, k]
#         surface_flux_values[i, j1, j2, direction, lower_left_element] = fstar_primary_lower_left[i, j1, j2, k]
#         surface_flux_values[i, j1, j2, direction, lower_right_element] = fstar_primary_lower_right[i, j1, j2, k]

#         # Use simple math expression to enhance the performance (against control flow), 
#         # it is equivalent to, `isequal(large_side, 1) * isequal(orientation, 1) * 2 +
#         #                       isequal(large_side, 1) * isequal(orientation, 2) * 4 +
#         #                       isequal(large_side, 1) * isequal(orientation, 3) * 6 +
#         #                       isequal(large_side, 2) * isequal(orientation, 1) * 1 +
#         #                       isequal(large_side, 2) * isequal(orientation, 2) * 3 +
#         #                       isequal(large_side, 2) * isequal(orientation, 3) * 5`.
#         # Please also check the original code in Trixi.jl when you modify this code.
#         direction = 2 * orientation - large_side + 1

#         @inbounds begin
#             for j1j1 in axes(reverse_upper, 2)
#                 tmp_upper_left[i, j1, j2, direction, large_element] += reverse_lower[j1, j1j1] *
#                                                                        fstar_secondary_upper_left[i, j1j1, j2, k]
#                 tmp_upper_right[i, j1, j2, direction, large_element] += reverse_upper[j1, j1j1] *
#                                                                         fstar_secondary_upper_right[i, j1j1, j2, k]
#                 tmp_lower_left[i, j1, j2, direction, large_element] += reverse_lower[j1, j1j1] *
#                                                                        fstar_secondary_lower_left[i, j1j1, j2, k]
#                 tmp_lower_right[i, j1, j2, direction, large_element] += reverse_upper[j1, j1j1] *
#                                                                         fstar_secondary_lower_right[i, j1j1, j2, k]
#             end
#         end
#     end

#     return nothing
# end

# # Kernel for copying mortar fluxes small to small and small to large - step 2
# function mortar_flux_copy_to_kernel!(surface_flux_values, tmp_surface_flux_values, tmp_upper_left,
#                                      tmp_upper_right, tmp_lower_left, tmp_lower_right,
#                                      reverse_upper, reverse_lower, neighbor_ids, large_sides,
#                                      orientations, equations::AbstractEquations{3})
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
#     k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

#     if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_values, 2)^2 &&
#         k <= length(orientations))
#         j1 = div(j - 1, size(surface_flux_values, 2)) + 1
#         j2 = rem(j - 1, size(surface_flux_values, 2)) + 1

#         large_element = neighbor_ids[5, k]

#         large_side = large_sides[k]
#         orientation = orientations[k]

#         # See step 1 for the explanation of the following expression
#         direction = 2 * orientation - large_side + 1

#         @inbounds begin
#             for j2j2 in axes(reverse_lower, 2)
#                 tmp_surface_flux_values[i, j1, j2, direction, large_element] += reverse_upper[j2, j2j2] *
#                                                                                 tmp_upper_left[i, j1, j2j2,
#                                                                                                direction,
#                                                                                                large_element]
#                 tmp_surface_flux_values[i, j1, j2, direction, large_element] += reverse_upper[j2, j2j2] *
#                                                                                 tmp_upper_right[i, j1, j2j2,
#                                                                                                 direction,
#                                                                                                 large_element]
#                 tmp_surface_flux_values[i, j1, j2, direction, large_element] += reverse_lower[j2, j2j2] *
#                                                                                 tmp_lower_left[i, j1, j2j2,
#                                                                                                direction,
#                                                                                                large_element]
#                 tmp_surface_flux_values[i, j1, j2, direction, large_element] += reverse_lower[j2, j2j2] *
#                                                                                 tmp_lower_right[i, j1, j2j2,
#                                                                                                 direction,
#                                                                                                 large_element]
#             end

#             surface_flux_values[i, j1, j2, direction, large_element] = tmp_surface_flux_values[i, j1, j2,
#                                                                                                direction,
#                                                                                                large_element]
#         end
#     end

#     return nothing
# end

# Kernel for copying mortar fluxes small to small and small to large (optimized)
function mortar_flux_copy_to_kernel!(surface_flux_values, tmp_surface_flux_values,
                                     tmp_upper_left, tmp_upper_right, tmp_lower_left, tmp_lower_right,
                                     fstar_primary_upper_left, fstar_primary_upper_right,
                                     fstar_primary_lower_left, fstar_primary_lower_right,
                                     fstar_secondary_upper_left, fstar_secondary_upper_right,
                                     fstar_secondary_lower_left, fstar_secondary_lower_right,
                                     reverse_upper, reverse_lower, neighbor_ids, large_sides,
                                     orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    # Loop stride for each dimension
    stride_x = gridDim().x * blockDim().x
    stride_y = gridDim().y * blockDim().y
    stride_z = gridDim().z * blockDim().z

    # Cooperative kernel needs stride loops to handle the constrained launch size
    while i <= size(surface_flux_values, 1)
        while j <= size(surface_flux_values, 2)^2
            while k <= length(orientations)
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

                    # Use simple math expression to enhance the performance (against control flow), 
                    # it is equivalent to, `isequal(large_side, 1) * isequal(orientation, 1) * 1 +
                    #                       isequal(large_side, 1) * isequal(orientation, 2) * 3 +
                    #                       isequal(large_side, 1) * isequal(orientation, 3) * 5 +
                    #                       isequal(large_side, 2) * isequal(orientation, 1) * 2 +
                    #                       isequal(large_side, 2) * isequal(orientation, 2) * 4 +
                    #                       isequal(large_side, 2) * isequal(orientation, 3) * 6`.
                    # Please also check the original code in Trixi.jl when you modify this code.
                    direction = 2 * orientation + large_side - 2

                    surface_flux_values[i, j1, j2, direction, upper_left_element] = fstar_primary_upper_left[i, j1, j2, k]
                    surface_flux_values[i, j1, j2, direction, upper_right_element] = fstar_primary_upper_right[i, j1, j2, k]
                    surface_flux_values[i, j1, j2, direction, lower_left_element] = fstar_primary_lower_left[i, j1, j2, k]
                    surface_flux_values[i, j1, j2, direction, lower_right_element] = fstar_primary_lower_right[i, j1, j2, k]

                    # Use simple math expression to enhance the performance (against control flow), 
                    # it is equivalent to, `isequal(large_side, 1) * isequal(orientation, 1) * 2 +
                    #                       isequal(large_side, 1) * isequal(orientation, 2) * 4 +
                    #                       isequal(large_side, 1) * isequal(orientation, 3) * 6 +
                    #                       isequal(large_side, 2) * isequal(orientation, 1) * 1 +
                    #                       isequal(large_side, 2) * isequal(orientation, 2) * 3 +
                    #                       isequal(large_side, 2) * isequal(orientation, 3) * 5`.
                    # Please also check the original code in Trixi.jl when you modify this code.
                    direction = 2 * orientation - large_side + 1
                end

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

                # Grid scope synchronization
                grid = CG.this_grid()
                CG.sync(grid)

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
                k += stride_z
            end
            j += stride_y
        end
        i += stride_x
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

# Functions that begin with `cuda_` are the functions that pack CUDA kernels together to do 
# partial work in semidiscretization. They are used to invoke kernels from the host (i.e., CPU) 
# and run them on the device (i.e., GPU).

# Note that `volume_integral::VolumeIntegralPureLGLFiniteVolume` is currently experimental
# in Trixi.jl and it is not implemented here.

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::DGSEM, cache)
    derivative_dhat = CuArray(dg.basis.derivative_dhat)

    # The maximum number of threads per block is the dominant factor when choosing the optimization 
    # method. However, there are other factors that may cause a launch failure, such as the maximum 
    # shared memory per block. Here, we have omitted all other factors, but this should be enhanced 
    # later for a safer kernel launch.

    # TODO: More checks before the kernel launch
    thread_per_block = size(du, 1) * size(du, 2)^3
    if thread_per_block > MAX_THREADS_PER_BLOCK
        # TODO: How to optimize when size is large
        flux_arr1 = similar(u)
        flux_arr2 = similar(u)
        flux_arr3 = similar(u)

        flux_kernel = @cuda launch=false flux_kernel!(flux_arr1, flux_arr2, flux_arr3, u, equations,
                                                      flux)
        flux_kernel(flux_arr1, flux_arr2, flux_arr3, u, equations, flux;
                    kernel_configurator_2d(flux_kernel, size(u, 2)^3, size(u, 5))...)

        weak_form_kernel = @cuda launch=false weak_form_kernel!(du, derivative_dhat, flux_arr1,
                                                                flux_arr2, flux_arr3)
        weak_form_kernel(du, derivative_dhat, flux_arr1, flux_arr2, flux_arr3;
                         kernel_configurator_3d(weak_form_kernel, size(du, 1), size(du, 2)^3,
                                                size(du, 5))...)
    else
        shmem_size = (size(du, 2)^2 + size(du, 1) * 3 * size(du, 2)^3) * sizeof(eltype(du))
        threads = (size(du, 1), size(du, 2)^3, 1)
        blocks = (1, 1, size(du, 5))
        @cuda threads=threads blocks=blocks shmem=shmem_size flux_weak_form_kernel!(du, u,
                                                                                    derivative_dhat,
                                                                                    equations,
                                                                                    flux)
    end

    return nothing
end

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM, cache)
    RealT = eltype(du)

    volume_flux = volume_integral.volume_flux

    # TODO: Move `set_diagonal_to_zero!` outside of loop and cache the result in DG on GPU
    derivative_split = dg.basis.derivative_split
    set_diagonal_to_zero!(derivative_split)
    derivative_split = CuArray(derivative_split)

    thread_per_block = size(du, 2)^3
    if thread_per_block > MAX_THREADS_PER_BLOCK
        # TODO: How to optimize when size is large
        volume_flux_arr1 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                          size(u, 2), size(u, 5))
        volume_flux_arr2 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                          size(u, 2), size(u, 5))
        volume_flux_arr3 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                          size(u, 2), size(u, 5))

        volume_flux_kernel = @cuda launch=false volume_flux_kernel!(volume_flux_arr1, volume_flux_arr2,
                                                                    volume_flux_arr3, u, equations,
                                                                    volume_flux)
        volume_flux_kernel(volume_flux_arr1, volume_flux_arr2, volume_flux_arr3, u, equations,
                           volume_flux;
                           kernel_configurator_2d(volume_flux_kernel, size(u, 2)^4, size(u, 5))...)

        volume_integral_kernel = @cuda launch=false volume_integral_kernel!(du, derivative_split,
                                                                            volume_flux_arr1,
                                                                            volume_flux_arr2,
                                                                            volume_flux_arr3, equations)
        volume_integral_kernel(du, derivative_split, volume_flux_arr1, volume_flux_arr2,
                               volume_flux_arr3, equations;
                               kernel_configurator_3d(volume_integral_kernel, size(du, 1),
                                                      size(du, 2)^3, size(du, 5))...)
    else
        shmem_size = (size(du, 2)^2 + size(du, 1) * size(du, 2)^3) * sizeof(RealT)
        threads = (1, size(du, 2)^3, 1)
        blocks = (1, 1, size(du, 5))
        @cuda threads=threads blocks=blocks shmem=shmem_size volume_flux_integral_kernel!(du, u,
                                                                                          derivative_split,
                                                                                          equations,
                                                                                          volume_flux)
    end

    return nothing
end

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms::True, equations,
                               volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM, cache)
    RealT = eltype(du)

    symmetric_flux, nonconservative_flux = dg.volume_integral.volume_flux

    # TODO: Move `set_diagonal_to_zero!` outside of loop and cache the result in DG on GPU
    derivative_split = dg.basis.derivative_split # create copy
    set_diagonal_to_zero!(derivative_split)
    derivative_split_zero = CuArray(derivative_split)
    derivative_split = CuArray(dg.basis.derivative_split)

    thread_per_block = size(du, 2)^3
    if thread_per_block > MAX_THREADS_PER_BLOCK
        # TODO: How to optimize when size is large
        symmetric_flux_arr1 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                             size(u, 2), size(u, 5))
        symmetric_flux_arr2 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                             size(u, 2), size(u, 5))
        symmetric_flux_arr3 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                             size(u, 2), size(u, 5))
        noncons_flux_arr1 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                           size(u, 2), size(u, 5))
        noncons_flux_arr2 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                           size(u, 2), size(u, 5))
        noncons_flux_arr3 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                           size(u, 2), size(u, 5))

        noncons_volume_flux_kernel = @cuda launch=false noncons_volume_flux_kernel!(symmetric_flux_arr1,
                                                                                    symmetric_flux_arr2,
                                                                                    symmetric_flux_arr3,
                                                                                    noncons_flux_arr1,
                                                                                    noncons_flux_arr2,
                                                                                    noncons_flux_arr3, u,
                                                                                    derivative_split_zero,
                                                                                    equations,
                                                                                    symmetric_flux,
                                                                                    nonconservative_flux)
        noncons_volume_flux_kernel(symmetric_flux_arr1, symmetric_flux_arr2, symmetric_flux_arr3,
                                   noncons_flux_arr1, noncons_flux_arr2, noncons_flux_arr3, u,
                                   derivative_split_zero, equations, symmetric_flux, nonconservative_flux;
                                   kernel_configurator_2d(noncons_volume_flux_kernel,
                                                          size(u, 2)^4, size(u, 5))...)

        volume_integral_kernel = @cuda launch=false volume_integral_kernel!(du, derivative_split,
                                                                            symmetric_flux_arr1,
                                                                            symmetric_flux_arr2,
                                                                            symmetric_flux_arr3,
                                                                            noncons_flux_arr1,
                                                                            noncons_flux_arr2,
                                                                            noncons_flux_arr3)
        volume_integral_kernel(du, derivative_split, symmetric_flux_arr1, symmetric_flux_arr2,
                               symmetric_flux_arr3, noncons_flux_arr1, noncons_flux_arr2,
                               noncons_flux_arr3;
                               kernel_configurator_3d(volume_integral_kernel, size(du, 1),
                                                      size(du, 2)^3, size(du, 5))...)
    else
        shmem_size = (size(du, 2)^2 * 2 + size(du, 1) * size(du, 2)^3) * sizeof(RealT)
        threads = (1, size(du, 2)^3, 1)
        blocks = (1, 1, size(du, 5))
        @cuda threads=threads blocks=blocks shmem=shmem_size noncons_volume_flux_integral_kernel!(du, u,
                                                                                                  derivative_split,
                                                                                                  derivative_split_zero,
                                                                                                  equations,
                                                                                                  symmetric_flux,
                                                                                                  nonconservative_flux)
    end

    return nothing
end

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralShockCapturingHG, dg::DGSEM, cache)
    RealT = eltype(du)

    volume_flux_dg, volume_flux_fv = dg.volume_integral.volume_flux_dg,
                                     dg.volume_integral.volume_flux_fv
    indicator = dg.volume_integral.indicator

    # TODO: Get copies of `u` and `du` on both device and host
    # FIXME: Scalar indexing on GPU arrays caused by using GPU cache
    alpha = indicator(Array(u), mesh, equations, dg, cache) # GPU cache
    alpha = CuArray(alpha)

    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))
    element_ids_dg = CUDA.zeros(Int, length(alpha))
    element_ids_dgfv = CUDA.zeros(Int, length(alpha))

    # See `pure_and_blended_element_ids!` in Trixi.jl (now deprecated)
    pure_blended_element_count_kernel = @cuda launch=false pure_blended_element_count_kernel!(element_ids_dg,
                                                                                              element_ids_dgfv,
                                                                                              alpha,
                                                                                              atol)
    pure_blended_element_count_kernel(element_ids_dg, element_ids_dgfv, alpha, atol;
                                      kernel_configurator_1d(pure_blended_element_count_kernel,
                                                             length(alpha))...)

    derivative_split = dg.basis.derivative_split
    set_diagonal_to_zero!(derivative_split) # temporarily set here, maybe move outside `rhs!`

    derivative_split = CuArray(derivative_split)
    inverse_weights = CuArray(dg.basis.inverse_weights)
    volume_flux_arr1 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                      size(u, 2), size(u, 5))
    volume_flux_arr2 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                      size(u, 2), size(u, 5))
    volume_flux_arr3 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                      size(u, 2), size(u, 5))

    fstar1_L = cache.fstar1_L
    fstar1_R = cache.fstar1_R
    fstar2_L = cache.fstar2_L
    fstar2_R = cache.fstar2_R
    fstar3_L = cache.fstar3_L
    fstar3_R = cache.fstar3_R

    volume_flux_dgfv_kernel = @cuda launch=false volume_flux_dgfv_kernel!(volume_flux_arr1,
                                                                          volume_flux_arr2,
                                                                          volume_flux_arr3,
                                                                          fstar1_L,
                                                                          fstar1_R, fstar2_L,
                                                                          fstar2_R,
                                                                          fstar3_L, fstar3_R, u,
                                                                          element_ids_dgfv,
                                                                          equations,
                                                                          volume_flux_dg,
                                                                          volume_flux_fv)
    volume_flux_dgfv_kernel(volume_flux_arr1, volume_flux_arr2, volume_flux_arr3, fstar1_L,
                            fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R, u, element_ids_dgfv,
                            equations, volume_flux_dg, volume_flux_fv;
                            kernel_configurator_2d(volume_flux_dgfv_kernel, size(u, 2)^4,
                                                   size(u, 5))...)

    volume_integral_dg_kernel = @cuda launch=false volume_integral_dg_kernel!(du, element_ids_dg,
                                                                              element_ids_dgfv,
                                                                              alpha,
                                                                              derivative_split,
                                                                              volume_flux_arr1,
                                                                              volume_flux_arr2,
                                                                              volume_flux_arr3,
                                                                              equations)
    volume_integral_dg_kernel(du, element_ids_dg, element_ids_dgfv, alpha, derivative_split,
                              volume_flux_arr1, volume_flux_arr2, volume_flux_arr3, equations;
                              kernel_configurator_3d(volume_integral_dg_kernel, size(du, 1),
                                                     size(du, 2)^3, size(du, 5))...)

    volume_integral_fv_kernel = @cuda launch=false volume_integral_fv_kernel!(du, fstar1_L,
                                                                              fstar1_R,
                                                                              fstar2_L, fstar2_R,
                                                                              fstar3_L, fstar3_R,
                                                                              inverse_weights,
                                                                              element_ids_dgfv,
                                                                              alpha)
    volume_integral_fv_kernel(du, fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                              inverse_weights, element_ids_dgfv, alpha;
                              kernel_configurator_2d(volume_integral_fv_kernel, size(u, 2)^3,
                                                     size(u, 5))...)

    return nothing
end

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms::True, equations,
                               volume_integral::VolumeIntegralShockCapturingHG, dg::DGSEM, cache)
    RealT = eltype(du)

    volume_flux_dg, nonconservative_flux_dg = dg.volume_integral.volume_flux_dg
    volume_flux_fv, nonconservative_flux_fv = dg.volume_integral.volume_flux_fv
    indicator = dg.volume_integral.indicator

    # TODO: Get copies of `u` and `du` on both device and host
    # FIXME: Scalar indexing on GPU arrays caused by using GPU cache
    alpha = indicator(Array(u), mesh, equations, dg, cache) # GPU cache
    alpha = CuArray(alpha)

    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))
    element_ids_dg = CUDA.zeros(Int, length(alpha))
    element_ids_dgfv = CUDA.zeros(Int, length(alpha))

    # See `pure_and_blended_element_ids!` in Trixi.jl (now deprecated)
    pure_blended_element_count_kernel = @cuda launch=false pure_blended_element_count_kernel!(element_ids_dg,
                                                                                              element_ids_dgfv,
                                                                                              alpha,
                                                                                              atol)
    pure_blended_element_count_kernel(element_ids_dg, element_ids_dgfv, alpha, atol;
                                      kernel_configurator_1d(pure_blended_element_count_kernel,
                                                             length(alpha))...)

    derivative_split = dg.basis.derivative_split
    set_diagonal_to_zero!(derivative_split) # temporarily set here, maybe move outside `rhs!`

    derivative_split = CuArray(derivative_split)
    inverse_weights = CuArray(dg.basis.inverse_weights)
    volume_flux_arr1 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                      size(u, 2), size(u, 5))
    volume_flux_arr2 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                      size(u, 2), size(u, 5))
    volume_flux_arr3 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                      size(u, 2), size(u, 5))
    noncons_flux_arr1 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                       size(u, 2), size(u, 5))
    noncons_flux_arr2 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                       size(u, 2), size(u, 5))
    noncons_flux_arr3 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                       size(u, 2), size(u, 5))

    fstar1_L = cache.fstar1_L
    fstar1_R = cache.fstar1_R
    fstar2_L = cache.fstar2_L
    fstar2_R = cache.fstar2_R
    fstar3_L = cache.fstar3_L
    fstar3_R = cache.fstar3_R

    volume_flux_dgfv_kernel = @cuda launch=false volume_flux_dgfv_kernel!(volume_flux_arr1,
                                                                          volume_flux_arr2,
                                                                          volume_flux_arr3,
                                                                          noncons_flux_arr1,
                                                                          noncons_flux_arr2,
                                                                          noncons_flux_arr3,
                                                                          fstar1_L, fstar1_R,
                                                                          fstar2_L, fstar2_R,
                                                                          fstar3_L, fstar3_R,
                                                                          u, element_ids_dgfv,
                                                                          derivative_split,
                                                                          equations,
                                                                          volume_flux_dg,
                                                                          nonconservative_flux_dg,
                                                                          volume_flux_fv,
                                                                          nonconservative_flux_fv)
    volume_flux_dgfv_kernel(volume_flux_arr1, volume_flux_arr2, volume_flux_arr3,
                            noncons_flux_arr1, noncons_flux_arr2, noncons_flux_arr3,
                            fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                            u, element_ids_dgfv, derivative_split, equations, volume_flux_dg,
                            nonconservative_flux_dg, volume_flux_fv, nonconservative_flux_fv;
                            kernel_configurator_2d(volume_flux_dgfv_kernel, size(u, 2)^4,
                                                   size(u, 5))...)

    derivative_split = CuArray(dg.basis.derivative_split) # use original `derivative_split`

    volume_integral_dg_kernel = @cuda launch=false volume_integral_dg_kernel!(du, element_ids_dg,
                                                                              element_ids_dgfv,
                                                                              alpha,
                                                                              derivative_split,
                                                                              volume_flux_arr1,
                                                                              volume_flux_arr2,
                                                                              volume_flux_arr3,
                                                                              noncons_flux_arr1,
                                                                              noncons_flux_arr2,
                                                                              noncons_flux_arr3,
                                                                              equations)
    volume_integral_dg_kernel(du, element_ids_dg, element_ids_dgfv, alpha, derivative_split,
                              volume_flux_arr1, volume_flux_arr2, volume_flux_arr3,
                              noncons_flux_arr1, noncons_flux_arr2, noncons_flux_arr3, equations;
                              kernel_configurator_3d(volume_integral_dg_kernel, size(du, 1),
                                                     size(du, 2)^3, size(du, 5))...)

    volume_integral_fv_kernel = @cuda launch=false volume_integral_fv_kernel!(du, fstar1_L,
                                                                              fstar1_R,
                                                                              fstar2_L, fstar2_R,
                                                                              fstar3_L, fstar3_R,
                                                                              inverse_weights,
                                                                              element_ids_dgfv,
                                                                              alpha)
    volume_integral_fv_kernel(du, fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                              inverse_weights, element_ids_dgfv, alpha;
                              kernel_configurator_2d(volume_integral_fv_kernel, size(u, 2)^3,
                                                     size(u, 5))...)

    return nothing
end

# Pack kernels to prolonging solution to interfaces
function cuda_prolong2interfaces!(u, mesh::TreeMesh{3}, equations, cache)
    neighbor_ids = cache.interfaces.neighbor_ids
    orientations = cache.interfaces.orientations
    interfaces_u = cache.interfaces.u

    prolong_interfaces_kernel = @cuda launch=false prolong_interfaces_kernel!(interfaces_u, u,
                                                                              neighbor_ids,
                                                                              orientations,
                                                                              equations)
    prolong_interfaces_kernel(interfaces_u, u, neighbor_ids, orientations, equations;
                              kernel_configurator_2d(prolong_interfaces_kernel,
                                                     size(interfaces_u, 2) *
                                                     size(interfaces_u, 3)^2,
                                                     size(interfaces_u, 5))...)

    return nothing
end

# Pack kernels for calculating interface fluxes
function cuda_interface_flux!(mesh::TreeMesh{3}, nonconservative_terms::False, equations, dg::DGSEM,
                              cache)
    RealT = eltype(cache.elements)

    surface_flux = dg.surface_integral.surface_flux

    neighbor_ids = cache.interfaces.neighbor_ids
    orientations = cache.interfaces.orientations
    interfaces_u = cache.interfaces.u
    surface_flux_values = cache.elements.surface_flux_values
    surface_flux_arr = CuArray{RealT}(undef, size(interfaces_u)[2:end]...)

    surface_flux_kernel = @cuda launch=false surface_flux_kernel!(surface_flux_arr, interfaces_u,
                                                                  orientations, equations,
                                                                  surface_flux)
    surface_flux_kernel(surface_flux_arr, interfaces_u, orientations, equations, surface_flux;
                        kernel_configurator_3d(surface_flux_kernel, size(interfaces_u, 3),
                                               size(interfaces_u, 4),
                                               size(interfaces_u, 5))...)

    interface_flux_kernel = @cuda launch=false interface_flux_kernel!(surface_flux_values,
                                                                      surface_flux_arr,
                                                                      neighbor_ids, orientations,
                                                                      equations)
    interface_flux_kernel(surface_flux_values, surface_flux_arr, neighbor_ids, orientations,
                          equations;
                          kernel_configurator_3d(interface_flux_kernel,
                                                 size(surface_flux_values, 1),
                                                 size(interfaces_u, 3)^2,
                                                 size(interfaces_u, 5))...)

    return nothing
end

# Pack kernels for calculating interface fluxes
function cuda_interface_flux!(mesh::TreeMesh{3}, nonconservative_terms::True, equations, dg::DGSEM,
                              cache)
    RealT = eltype(cache.elements)

    surface_flux, nonconservative_flux = dg.surface_integral.surface_flux

    neighbor_ids = cache.interfaces.neighbor_ids
    orientations = cache.interfaces.orientations
    interfaces_u = cache.interfaces.u
    surface_flux_values = cache.elements.surface_flux_values

    surface_flux_arr = CuArray{RealT}(undef, size(interfaces_u)[2:end]...)
    noncons_left_arr = CuArray{RealT}(undef, size(interfaces_u)[2:end]...)
    noncons_right_arr = CuArray{RealT}(undef, size(interfaces_u)[2:end]...)

    surface_noncons_flux_kernel = @cuda launch=false surface_noncons_flux_kernel!(surface_flux_arr,
                                                                                  noncons_left_arr,
                                                                                  noncons_right_arr,
                                                                                  interfaces_u,
                                                                                  orientations,
                                                                                  equations,
                                                                                  surface_flux,
                                                                                  nonconservative_flux)
    surface_noncons_flux_kernel(surface_flux_arr, noncons_left_arr, noncons_right_arr, interfaces_u,
                                orientations, equations, surface_flux, nonconservative_flux;
                                kernel_configurator_3d(surface_noncons_flux_kernel,
                                                       size(interfaces_u, 3),
                                                       size(interfaces_u, 4),
                                                       size(interfaces_u, 5))...)

    interface_flux_kernel = @cuda launch=false interface_flux_kernel!(surface_flux_values,
                                                                      surface_flux_arr,
                                                                      noncons_left_arr,
                                                                      noncons_right_arr,
                                                                      neighbor_ids, orientations,
                                                                      equations)
    interface_flux_kernel(surface_flux_values, surface_flux_arr, noncons_left_arr,
                          noncons_right_arr,
                          neighbor_ids, orientations, equations;
                          kernel_configurator_3d(interface_flux_kernel,
                                                 size(surface_flux_values, 1),
                                                 size(interfaces_u, 3)^2,
                                                 size(interfaces_u, 5))...)

    return nothing
end

# Dummy function for asserting boundaries
function cuda_prolong2boundaries!(u, mesh::TreeMesh{3},
                                  boundary_condition::BoundaryConditionPeriodic, equations, cache)
    @assert iszero(length(cache.boundaries.orientations))
end

# Pack kernels for prolonging solution to boundaries
function cuda_prolong2boundaries!(u, mesh::TreeMesh{3}, boundary_conditions::NamedTuple, equations,
                                  cache)
    neighbor_ids = cache.boundaries.neighbor_ids
    neighbor_sides = cache.boundaries.neighbor_sides
    orientations = cache.boundaries.orientations
    boundaries_u = cache.boundaries.u

    prolong_boundaries_kernel = @cuda launch=false prolong_boundaries_kernel!(boundaries_u, u,
                                                                              neighbor_ids,
                                                                              neighbor_sides,
                                                                              orientations,
                                                                              equations)
    prolong_boundaries_kernel(boundaries_u, u, neighbor_ids, neighbor_sides, orientations,
                              equations;
                              kernel_configurator_2d(prolong_boundaries_kernel,
                                                     size(boundaries_u, 2) *
                                                     size(boundaries_u, 3)^2,
                                                     size(boundaries_u, 5))...)

    return nothing
end

# Dummy function for asserting boundary fluxes
function cuda_boundary_flux!(t, mesh::TreeMesh{3}, boundary_condition::BoundaryConditionPeriodic,
                             nonconservative_terms, equations, dg::DGSEM, cache)
    @assert iszero(length(cache.boundaries.orientations))
end

# Pack kernels for calculating boundary fluxes
function cuda_boundary_flux!(t, mesh::TreeMesh{3}, boundary_conditions::NamedTuple,
                             nonconservative_terms, equations, dg::DGSEM, cache)
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
                                                                    equations, surface_flux)
    boundary_flux_kernel(surface_flux_values, boundaries_u, node_coordinates, t, boundary_arr,
                         indices_arr, neighbor_ids, neighbor_sides, orientations,
                         boundary_conditions_callable, equations, surface_flux;
                         kernel_configurator_2d(boundary_flux_kernel,
                                                size(surface_flux_values, 2)^2,
                                                length(boundary_arr))...)

    return nothing
end

# Dummy function for asserting mortars 
function cuda_prolong2mortars!(u, mesh::TreeMesh{3}, cache_mortars::False, dg::DGSEM, cache)
    @assert iszero(length(cache.mortars.orientations))
end

# # Pack kernels for prolonging solution to mortars
# function cuda_prolong2mortars!(u, mesh::TreeMesh{3}, cache_mortars::True, dg::DGSEM, cache)
#     neighbor_ids = cache.mortars.neighbor_ids
#     large_sides = cache.mortars.large_sides
#     orientations = cache.mortars.orientations

#     # The original CPU arrays hold NaNs
#     u_upper_left = cache.mortars.u_upper_left
#     u_upper_right = cache.mortars.u_upper_right
#     u_lower_left = cache.mortars.u_lower_left
#     u_lower_right = cache.mortars.u_lower_right
#     forward_upper = CuArray(dg.mortar.forward_upper)
#     forward_lower = CuArray(dg.mortar.forward_lower)

#     prolong_mortars_small2small_kernel = @cuda launch=false prolong_mortars_small2small_kernel!(u_upper_left,
#                                                                                                 u_upper_right,
#                                                                                                 u_lower_left,
#                                                                                                 u_lower_right,
#                                                                                                 u,
#                                                                                                 neighbor_ids,
#                                                                                                 large_sides,
#                                                                                                 orientations)
#     prolong_mortars_small2small_kernel(u_upper_left, u_upper_right, u_lower_left, u_lower_right, u,
#                                        neighbor_ids, large_sides, orientations;
#                                        kernel_configurator_3d(prolong_mortars_small2small_kernel,
#                                                               size(u_upper_left, 2),
#                                                               size(u_upper_left, 3)^2,
#                                                               size(u_upper_left, 5))...)

#     tmp_upper_left = zero(similar(u_upper_left)) # undef to zero
#     tmp_upper_right = zero(similar(u_upper_right)) # undef to zero
#     tmp_lower_left = zero(similar(u_lower_left)) # undef to zero
#     tmp_lower_right = zero(similar(u_lower_right)) # undef to zero

#     prolong_mortars_large2small_kernel = @cuda launch=false prolong_mortars_large2small_kernel!(tmp_upper_left,
#                                                                                                 tmp_upper_right,
#                                                                                                 tmp_lower_left,
#                                                                                                 tmp_lower_right,
#                                                                                                 u,
#                                                                                                 forward_upper,
#                                                                                                 forward_lower,
#                                                                                                 neighbor_ids,
#                                                                                                 large_sides,
#                                                                                                 orientations)
#     prolong_mortars_large2small_kernel(tmp_upper_left, tmp_upper_right, tmp_lower_left,
#                                        tmp_lower_right, u, forward_upper, forward_lower,
#                                        neighbor_ids, large_sides, orientations;
#                                        kernel_configurator_3d(prolong_mortars_large2small_kernel,
#                                                               size(u_upper_left, 2),
#                                                               size(u_upper_left, 3)^2,
#                                                               size(u_upper_left, 5))...)

#     prolong_mortars_large2small_kernel = @cuda launch=false prolong_mortars_large2small_kernel!(u_upper_left,
#                                                                                                 u_upper_right,
#                                                                                                 u_lower_left,
#                                                                                                 u_lower_right,
#                                                                                                 tmp_upper_left,
#                                                                                                 tmp_upper_right,
#                                                                                                 tmp_lower_left,
#                                                                                                 tmp_lower_right,
#                                                                                                 forward_upper,
#                                                                                                 forward_lower,
#                                                                                                 large_sides)
#     prolong_mortars_large2small_kernel(u_upper_left, u_upper_right, u_lower_left, u_lower_right,
#                                        tmp_upper_left, tmp_upper_right, tmp_lower_left,
#                                        tmp_lower_right, forward_upper, forward_lower, large_sides;
#                                        kernel_configurator_3d(prolong_mortars_large2small_kernel,
#                                                               size(u_upper_left, 2),
#                                                               size(u_upper_left, 3)^2,
#                                                               size(u_upper_left, 5))...)

#     return nothing
# end

# Pack kernels for prolonging solution to mortars (optimized)
function cuda_prolong2mortars!(u, mesh::TreeMesh{3}, cache_mortars::True, dg::DGSEM, cache)
    neighbor_ids = cache.mortars.neighbor_ids
    large_sides = cache.mortars.large_sides
    orientations = cache.mortars.orientations

    # The original CPU arrays hold NaNs
    u_upper_left = cache.mortars.u_upper_left
    u_upper_right = cache.mortars.u_upper_right
    u_lower_left = cache.mortars.u_lower_left
    u_lower_right = cache.mortars.u_lower_right
    forward_upper = CuArray(dg.mortar.forward_upper)
    forward_lower = CuArray(dg.mortar.forward_lower)

    prolong_mortars_small2small_kernel = @cuda launch=false prolong_mortars_small2small_kernel!(u_upper_left,
                                                                                                u_upper_right,
                                                                                                u_lower_left,
                                                                                                u_lower_right,
                                                                                                u,
                                                                                                neighbor_ids,
                                                                                                large_sides,
                                                                                                orientations)
    prolong_mortars_small2small_kernel(u_upper_left, u_upper_right, u_lower_left, u_lower_right, u,
                                       neighbor_ids, large_sides, orientations;
                                       kernel_configurator_3d(prolong_mortars_small2small_kernel,
                                                              size(u_upper_left, 2),
                                                              size(u_upper_left, 3)^2,
                                                              size(u_upper_left, 5))...)

    tmp_upper_left = zero(similar(u_upper_left)) # undef to zero
    tmp_upper_right = zero(similar(u_upper_right)) # undef to zero
    tmp_lower_left = zero(similar(u_lower_left)) # undef to zero
    tmp_lower_right = zero(similar(u_lower_right)) # undef to zero

    prolong_mortars_large2small_kernel = @cuda launch=false prolong_mortars_large2small_kernel!(u_upper_left,
                                                                                                u_upper_right,
                                                                                                u_lower_left,
                                                                                                u_lower_right,
                                                                                                tmp_upper_left,
                                                                                                tmp_upper_right,
                                                                                                tmp_lower_left,
                                                                                                tmp_lower_right,
                                                                                                u, forward_upper,
                                                                                                forward_lower,
                                                                                                neighbor_ids,
                                                                                                large_sides,
                                                                                                orientations)
    prolong_mortars_large2small_kernel(u_upper_left, u_upper_right, u_lower_left, u_lower_right,
                                       tmp_upper_left, tmp_upper_right, tmp_lower_left,
                                       tmp_lower_right, u, forward_upper, forward_lower, neighbor_ids,
                                       large_sides, orientations; cooperative = true,
                                       kernel_configurator_coop_3d(prolong_mortars_large2small_kernel,
                                                                   size(u_upper_left, 2),
                                                                   size(u_upper_left, 3)^2,
                                                                   size(u_upper_left, 5))...)

    return nothing
end

# Dummy function for asserting mortar fluxes
function cuda_mortar_flux!(mesh::TreeMesh{3}, cache_mortars::False, nonconservative_terms,
                           equations, dg::DGSEM, cache)
    @assert iszero(length(cache.mortars.orientations))
end

# # Pack kernels for calculating mortar fluxes
# function cuda_mortar_flux!(mesh::TreeMesh{3}, cache_mortars::True, nonconservative_terms::False,
#                            equations, dg::DGSEM, cache)
#     surface_flux = dg.surface_integral.surface_flux

#     neighbor_ids = cache.mortars.neighbor_ids
#     large_sides = cache.mortars.large_sides
#     orientations = cache.mortars.orientations

#     # The original CPU arrays hold NaNs
#     u_upper_left = cache.mortars.u_upper_left
#     u_upper_right = cache.mortars.u_upper_right
#     u_lower_left = cache.mortars.u_lower_left
#     u_lower_right = cache.mortars.u_lower_right
#     reverse_upper = CuArray(dg.mortar.reverse_upper)
#     reverse_lower = CuArray(dg.mortar.reverse_lower)

#     surface_flux_values = cache.elements.surface_flux_values
#     tmp_surface_flux_values = zero(similar(surface_flux_values)) # undef to zero

#     fstar_primary_upper_left = cache.fstar_primary_upper_left
#     fstar_primary_upper_right = cache.fstar_primary_upper_right
#     fstar_primary_lower_left = cache.fstar_primary_lower_left
#     fstar_primary_lower_right = cache.fstar_primary_lower_right
#     fstar_secondary_upper_left = cache.fstar_secondary_upper_left
#     fstar_secondary_upper_right = cache.fstar_secondary_upper_right
#     fstar_secondary_lower_left = cache.fstar_secondary_lower_left
#     fstar_secondary_lower_right = cache.fstar_secondary_lower_right

#     mortar_flux_kernel = @cuda launch=false mortar_flux_kernel!(fstar_primary_upper_left,
#                                                                 fstar_primary_upper_right,
#                                                                 fstar_primary_lower_left,
#                                                                 fstar_primary_lower_right,
#                                                                 fstar_secondary_upper_left,
#                                                                 fstar_secondary_upper_right,
#                                                                 fstar_secondary_lower_left,
#                                                                 fstar_secondary_lower_right,
#                                                                 u_upper_left, u_upper_right,
#                                                                 u_lower_left, u_lower_right,
#                                                                 orientations, equations,
#                                                                 surface_flux)
#     mortar_flux_kernel(fstar_primary_upper_left, fstar_primary_upper_right,
#                        fstar_primary_lower_left, fstar_primary_lower_right,
#                        fstar_secondary_upper_left, fstar_secondary_upper_right,
#                        fstar_secondary_lower_left, fstar_secondary_lower_right,
#                        u_upper_left, u_upper_right, u_lower_left, u_lower_right, orientations,
#                        equations, surface_flux;
#                        kernel_configurator_3d(mortar_flux_kernel, size(u_upper_left, 3),
#                                               size(u_upper_left, 4),
#                                               length(orientations))...)

#     tmp_upper_left = zero(similar(surface_flux_values)) # undef to zero
#     tmp_upper_right = zero(similar(surface_flux_values)) # undef to zero
#     tmp_lower_left = zero(similar(surface_flux_values)) # undef to zero
#     tmp_lower_right = zero(similar(surface_flux_values)) # undef to zero

#     mortar_flux_copy_to_kernel = @cuda launch=false mortar_flux_copy_to_kernel!(surface_flux_values,
#                                                                                 tmp_upper_left,
#                                                                                 tmp_upper_right,
#                                                                                 tmp_lower_left,
#                                                                                 tmp_lower_right,
#                                                                                 fstar_primary_upper_left,
#                                                                                 fstar_primary_upper_right,
#                                                                                 fstar_primary_lower_left,
#                                                                                 fstar_primary_lower_right,
#                                                                                 fstar_secondary_upper_left,
#                                                                                 fstar_secondary_upper_right,
#                                                                                 fstar_secondary_lower_left,
#                                                                                 fstar_secondary_lower_right,
#                                                                                 reverse_upper,
#                                                                                 reverse_lower,
#                                                                                 neighbor_ids,
#                                                                                 large_sides,
#                                                                                 orientations)
#     mortar_flux_copy_to_kernel(surface_flux_values, tmp_upper_left, tmp_upper_right, tmp_lower_left,
#                                tmp_lower_right, fstar_primary_upper_left, fstar_primary_upper_right,
#                                fstar_primary_lower_left, fstar_primary_lower_right,
#                                fstar_secondary_upper_left, fstar_secondary_upper_right,
#                                fstar_secondary_lower_left, fstar_secondary_lower_right,
#                                reverse_upper, reverse_lower, neighbor_ids, large_sides,
#                                orientations;
#                                kernel_configurator_3d(mortar_flux_copy_to_kernel,
#                                                       size(surface_flux_values, 1),
#                                                       size(surface_flux_values, 2)^2,
#                                                       length(orientations))...)

#     mortar_flux_copy_to_kernel = @cuda launch=false mortar_flux_copy_to_kernel!(surface_flux_values,
#                                                                                 tmp_surface_flux_values,
#                                                                                 tmp_upper_left,
#                                                                                 tmp_upper_right,
#                                                                                 tmp_lower_left,
#                                                                                 tmp_lower_right,
#                                                                                 reverse_upper,
#                                                                                 reverse_lower,
#                                                                                 neighbor_ids,
#                                                                                 large_sides,
#                                                                                 orientations,
#                                                                                 equations)
#     mortar_flux_copy_to_kernel(surface_flux_values, tmp_surface_flux_values, tmp_upper_left,
#                                tmp_upper_right, tmp_lower_left, tmp_lower_right, reverse_upper,
#                                reverse_lower, neighbor_ids, large_sides, orientations, equations;
#                                kernel_configurator_3d(mortar_flux_copy_to_kernel,
#                                                       size(surface_flux_values, 1),
#                                                       size(surface_flux_values, 2)^2,
#                                                       length(orientations))...)

#     return nothing
# end

# Pack kernels for calculating mortar fluxes (optimized)
function cuda_mortar_flux!(mesh::TreeMesh{3}, cache_mortars::True, nonconservative_terms::False,
                           equations, dg::DGSEM, cache)
    surface_flux = dg.surface_integral.surface_flux

    neighbor_ids = cache.mortars.neighbor_ids
    large_sides = cache.mortars.large_sides
    orientations = cache.mortars.orientations

    # The original CPU arrays hold NaNs
    u_upper_left = cache.mortars.u_upper_left
    u_upper_right = cache.mortars.u_upper_right
    u_lower_left = cache.mortars.u_lower_left
    u_lower_right = cache.mortars.u_lower_right
    reverse_upper = CuArray(dg.mortar.reverse_upper)
    reverse_lower = CuArray(dg.mortar.reverse_lower)

    surface_flux_values = cache.elements.surface_flux_values
    tmp_surface_flux_values = zero(similar(surface_flux_values)) # undef to zero

    fstar_primary_upper_left = cache.fstar_primary_upper_left
    fstar_primary_upper_right = cache.fstar_primary_upper_right
    fstar_primary_lower_left = cache.fstar_primary_lower_left
    fstar_primary_lower_right = cache.fstar_primary_lower_right
    fstar_secondary_upper_left = cache.fstar_secondary_upper_left
    fstar_secondary_upper_right = cache.fstar_secondary_upper_right
    fstar_secondary_lower_left = cache.fstar_secondary_lower_left
    fstar_secondary_lower_right = cache.fstar_secondary_lower_right

    mortar_flux_kernel = @cuda launch=false mortar_flux_kernel!(fstar_primary_upper_left,
                                                                fstar_primary_upper_right,
                                                                fstar_primary_lower_left,
                                                                fstar_primary_lower_right,
                                                                fstar_secondary_upper_left,
                                                                fstar_secondary_upper_right,
                                                                fstar_secondary_lower_left,
                                                                fstar_secondary_lower_right,
                                                                u_upper_left, u_upper_right,
                                                                u_lower_left, u_lower_right,
                                                                orientations, equations,
                                                                surface_flux)
    mortar_flux_kernel(fstar_primary_upper_left, fstar_primary_upper_right,
                       fstar_primary_lower_left, fstar_primary_lower_right,
                       fstar_secondary_upper_left, fstar_secondary_upper_right,
                       fstar_secondary_lower_left, fstar_secondary_lower_right,
                       u_upper_left, u_upper_right, u_lower_left, u_lower_right, orientations,
                       equations, surface_flux;
                       kernel_configurator_3d(mortar_flux_kernel, size(u_upper_left, 3),
                                              size(u_upper_left, 4),
                                              length(orientations))...)

    tmp_upper_left = zero(similar(surface_flux_values)) # undef to zero
    tmp_upper_right = zero(similar(surface_flux_values)) # undef to zero
    tmp_lower_left = zero(similar(surface_flux_values)) # undef to zero
    tmp_lower_right = zero(similar(surface_flux_values)) # undef to zero

    mortar_flux_copy_to_kernel = @cuda launch=false mortar_flux_copy_to_kernel!(surface_flux_values,
                                                                                tmp_surface_flux_values,
                                                                                tmp_upper_left,
                                                                                tmp_upper_right,
                                                                                tmp_lower_left,
                                                                                tmp_lower_right,
                                                                                fstar_primary_upper_left,
                                                                                fstar_primary_upper_right,
                                                                                fstar_primary_lower_left,
                                                                                fstar_primary_lower_right,
                                                                                fstar_secondary_upper_left,
                                                                                fstar_secondary_upper_right,
                                                                                fstar_secondary_lower_left,
                                                                                fstar_secondary_lower_right,
                                                                                reverse_upper,
                                                                                reverse_lower,
                                                                                neighbor_ids,
                                                                                large_sides,
                                                                                orientations)
    mortar_flux_copy_to_kernel(surface_flux_values, tmp_surface_flux_values, tmp_upper_left, tmp_upper_right,
                               tmp_lower_left, tmp_lower_right, fstar_primary_upper_left, fstar_primary_upper_right,
                               fstar_primary_lower_left, fstar_primary_lower_right, fstar_secondary_upper_left,
                               fstar_secondary_upper_right, fstar_secondary_lower_left, fstar_secondary_lower_right,
                               reverse_upper, reverse_lower, neighbor_ids, large_sides, orientations;
                               cooperative = true,
                               kernel_configurator_coop_3d(mortar_flux_copy_to_kernel,
                                                           size(surface_flux_values, 1),
                                                           size(surface_flux_values, 2)^2,
                                                           length(orientations))...)

    return nothing
end

# # Pack kernels for calculating mortar fluxes
# function cuda_mortar_flux!(mesh::TreeMesh{3}, cache_mortars::True, nonconservative_terms::True,
#                            equations, dg::DGSEM, cache)
#     surface_flux, nonconservative_flux = dg.surface_integral.surface_flux

#     neighbor_ids = cache.mortars.neighbor_ids
#     large_sides = cache.mortars.large_sides
#     orientations = cache.mortars.orientations

#     # The original CPU arrays hold NaNs
#     u_upper_left = cache.mortars.u_upper_left
#     u_upper_right = cache.mortars.u_upper_right
#     u_lower_left = cache.mortars.u_lower_left
#     u_lower_right = cache.mortars.u_lower_right
#     reverse_upper = CuArray(dg.mortar.reverse_upper)
#     reverse_lower = CuArray(dg.mortar.reverse_lower)

#     surface_flux_values = cache.elements.surface_flux_values
#     tmp_surface_flux_values = zero(similar(surface_flux_values)) # undef to zero

#     fstar_primary_upper_left = cache.fstar_primary_upper_left
#     fstar_primary_upper_right = cache.fstar_primary_upper_right
#     fstar_primary_lower_left = cache.fstar_primary_lower_left
#     fstar_primary_lower_right = cache.fstar_primary_lower_right
#     fstar_secondary_upper_left = cache.fstar_secondary_upper_left
#     fstar_secondary_upper_right = cache.fstar_secondary_upper_right
#     fstar_secondary_lower_left = cache.fstar_secondary_lower_left
#     fstar_secondary_lower_right = cache.fstar_secondary_lower_right

#     mortar_flux_kernel = @cuda launch=false mortar_flux_kernel!(fstar_primary_upper_left,
#                                                                 fstar_primary_upper_right,
#                                                                 fstar_primary_lower_left,
#                                                                 fstar_primary_lower_right,
#                                                                 fstar_secondary_upper_left,
#                                                                 fstar_secondary_upper_right,
#                                                                 fstar_secondary_lower_left,
#                                                                 fstar_secondary_lower_right,
#                                                                 u_upper_left, u_upper_right,
#                                                                 u_lower_left, u_lower_right,
#                                                                 orientations, large_sides,
#                                                                 equations, surface_flux,
#                                                                 nonconservative_flux)
#     mortar_flux_kernel(fstar_primary_upper_left, fstar_primary_upper_right,
#                        fstar_primary_lower_left, fstar_primary_lower_right,
#                        fstar_secondary_upper_left, fstar_secondary_upper_right,
#                        fstar_secondary_lower_left, fstar_secondary_lower_right,
#                        u_upper_left, u_upper_right, u_lower_left, u_lower_right, orientations,
#                        large_sides, equations, surface_flux, nonconservative_flux;
#                        kernel_configurator_3d(mortar_flux_kernel, size(u_upper_left, 3),
#                                               size(u_upper_left, 4),
#                                               length(orientations))...)

#     tmp_upper_left = zero(similar(surface_flux_values)) # undef to zero
#     tmp_upper_right = zero(similar(surface_flux_values)) # undef to zero
#     tmp_lower_left = zero(similar(surface_flux_values)) # undef to zero
#     tmp_lower_right = zero(similar(surface_flux_values)) # undef to zero

#     mortar_flux_copy_to_kernel = @cuda launch=false mortar_flux_copy_to_kernel!(surface_flux_values,
#                                                                                 tmp_upper_left,
#                                                                                 tmp_upper_right,
#                                                                                 tmp_lower_left,
#                                                                                 tmp_lower_right,
#                                                                                 fstar_primary_upper_left,
#                                                                                 fstar_primary_upper_right,
#                                                                                 fstar_primary_lower_left,
#                                                                                 fstar_primary_lower_right,
#                                                                                 fstar_secondary_upper_left,
#                                                                                 fstar_secondary_upper_right,
#                                                                                 fstar_secondary_lower_left,
#                                                                                 fstar_secondary_lower_right,
#                                                                                 reverse_upper,
#                                                                                 reverse_lower,
#                                                                                 neighbor_ids,
#                                                                                 large_sides,
#                                                                                 orientations)
#     mortar_flux_copy_to_kernel(surface_flux_values, tmp_upper_left, tmp_upper_right, tmp_lower_left,
#                                tmp_lower_right, fstar_primary_upper_left, fstar_primary_upper_right,
#                                fstar_primary_lower_left, fstar_primary_lower_right,
#                                fstar_secondary_upper_left, fstar_secondary_upper_right,
#                                fstar_secondary_lower_left, fstar_secondary_lower_right,
#                                reverse_upper, reverse_lower, neighbor_ids, large_sides,
#                                orientations;
#                                kernel_configurator_3d(mortar_flux_copy_to_kernel,
#                                                       size(surface_flux_values, 1),
#                                                       size(surface_flux_values, 2)^2,
#                                                       length(orientations))...)

#     mortar_flux_copy_to_kernel = @cuda launch=false mortar_flux_copy_to_kernel!(surface_flux_values,
#                                                                                 tmp_surface_flux_values,
#                                                                                 tmp_upper_left,
#                                                                                 tmp_upper_right,
#                                                                                 tmp_lower_left,
#                                                                                 tmp_lower_right,
#                                                                                 reverse_upper,
#                                                                                 reverse_lower,
#                                                                                 neighbor_ids,
#                                                                                 large_sides,
#                                                                                 orientations,
#                                                                                 equations)
#     mortar_flux_copy_to_kernel(surface_flux_values, tmp_surface_flux_values, tmp_upper_left,
#                                tmp_upper_right, tmp_lower_left, tmp_lower_right, reverse_upper,
#                                reverse_lower, neighbor_ids, large_sides, orientations, equations;
#                                kernel_configurator_3d(mortar_flux_copy_to_kernel,
#                                                       size(surface_flux_values, 1),
#                                                       size(surface_flux_values, 2)^2,
#                                                       length(orientations))...)

#     return nothing
# end

# Pack kernels for calculating mortar fluxes (optimized)
function cuda_mortar_flux!(mesh::TreeMesh{3}, cache_mortars::True, nonconservative_terms::True,
                           equations, dg::DGSEM, cache)
    surface_flux, nonconservative_flux = dg.surface_integral.surface_flux

    neighbor_ids = cache.mortars.neighbor_ids
    large_sides = cache.mortars.large_sides
    orientations = cache.mortars.orientations

    # The original CPU arrays hold NaNs
    u_upper_left = cache.mortars.u_upper_left
    u_upper_right = cache.mortars.u_upper_right
    u_lower_left = cache.mortars.u_lower_left
    u_lower_right = cache.mortars.u_lower_right
    reverse_upper = CuArray(dg.mortar.reverse_upper)
    reverse_lower = CuArray(dg.mortar.reverse_lower)

    surface_flux_values = cache.elements.surface_flux_values
    tmp_surface_flux_values = zero(similar(surface_flux_values)) # undef to zero

    fstar_primary_upper_left = cache.fstar_primary_upper_left
    fstar_primary_upper_right = cache.fstar_primary_upper_right
    fstar_primary_lower_left = cache.fstar_primary_lower_left
    fstar_primary_lower_right = cache.fstar_primary_lower_right
    fstar_secondary_upper_left = cache.fstar_secondary_upper_left
    fstar_secondary_upper_right = cache.fstar_secondary_upper_right
    fstar_secondary_lower_left = cache.fstar_secondary_lower_left
    fstar_secondary_lower_right = cache.fstar_secondary_lower_right

    mortar_flux_kernel = @cuda launch=false mortar_flux_kernel!(fstar_primary_upper_left,
                                                                fstar_primary_upper_right,
                                                                fstar_primary_lower_left,
                                                                fstar_primary_lower_right,
                                                                fstar_secondary_upper_left,
                                                                fstar_secondary_upper_right,
                                                                fstar_secondary_lower_left,
                                                                fstar_secondary_lower_right,
                                                                u_upper_left, u_upper_right,
                                                                u_lower_left, u_lower_right,
                                                                orientations, large_sides,
                                                                equations, surface_flux,
                                                                nonconservative_flux)
    mortar_flux_kernel(fstar_primary_upper_left, fstar_primary_upper_right,
                       fstar_primary_lower_left, fstar_primary_lower_right,
                       fstar_secondary_upper_left, fstar_secondary_upper_right,
                       fstar_secondary_lower_left, fstar_secondary_lower_right,
                       u_upper_left, u_upper_right, u_lower_left, u_lower_right, orientations,
                       large_sides, equations, surface_flux, nonconservative_flux;
                       kernel_configurator_3d(mortar_flux_kernel, size(u_upper_left, 3),
                                              size(u_upper_left, 4),
                                              length(orientations))...)

    tmp_upper_left = zero(similar(surface_flux_values)) # undef to zero
    tmp_upper_right = zero(similar(surface_flux_values)) # undef to zero
    tmp_lower_left = zero(similar(surface_flux_values)) # undef to zero
    tmp_lower_right = zero(similar(surface_flux_values)) # undef to zero

    mortar_flux_copy_to_kernel = @cuda launch=false mortar_flux_copy_to_kernel!(surface_flux_values,
                                                                                tmp_surface_flux_values,
                                                                                tmp_upper_left,
                                                                                tmp_upper_right,
                                                                                tmp_lower_left,
                                                                                tmp_lower_right,
                                                                                fstar_primary_upper_left,
                                                                                fstar_primary_upper_right,
                                                                                fstar_primary_lower_left,
                                                                                fstar_primary_lower_right,
                                                                                fstar_secondary_upper_left,
                                                                                fstar_secondary_upper_right,
                                                                                fstar_secondary_lower_left,
                                                                                fstar_secondary_lower_right,
                                                                                reverse_upper,
                                                                                reverse_lower,
                                                                                neighbor_ids,
                                                                                large_sides,
                                                                                orientations)
    mortar_flux_copy_to_kernel(surface_flux_values, tmp_surface_flux_values, tmp_upper_left, tmp_upper_right,
                               tmp_lower_left, tmp_lower_right, fstar_primary_upper_left, fstar_primary_upper_right,
                               fstar_primary_lower_left, fstar_primary_lower_right, fstar_secondary_upper_left,
                               fstar_secondary_upper_right, fstar_secondary_lower_left, fstar_secondary_lower_right,
                               reverse_upper, reverse_lower, neighbor_ids, large_sides, orientations;
                               cooperative = true,
                               kernel_configurator_coop_3d(mortar_flux_copy_to_kernel,
                                                           size(surface_flux_values, 1),
                                                           size(surface_flux_values, 2)^2,
                                                           length(orientations))...)

    return nothing
end

# Pack kernels for calculating surface integrals
function cuda_surface_integral!(du, mesh::TreeMesh{3}, equations, dg::DGSEM, cache)
    factor_arr = CuArray([
                             dg.basis.boundary_interpolation[1, 1],
                             dg.basis.boundary_interpolation[size(du, 2), 2]
                         ])
    surface_flux_values = cache.elements.surface_flux_values

    surface_integral_kernel = @cuda launch=false surface_integral_kernel!(du, factor_arr,
                                                                          surface_flux_values,
                                                                          equations)
    surface_integral_kernel(du, factor_arr, surface_flux_values, equations;
                            kernel_configurator_3d(surface_integral_kernel, size(du, 1),
                                                   size(du, 2)^3, size(du, 5))...)

    return nothing
end

# Pack kernels for applying Jacobian to reference element
function cuda_jacobian!(du, mesh::TreeMesh{3}, equations, cache)
    inverse_jacobian = cache.elements.inverse_jacobian

    jacobian_kernel = @cuda launch=false jacobian_kernel!(du, inverse_jacobian, equations)
    jacobian_kernel(du, inverse_jacobian, equations;
                    kernel_configurator_3d(jacobian_kernel, size(du, 1), size(du, 2)^3,
                                           size(du, 5))...)

    return nothing
end

# Dummy function returning nothing            
function cuda_sources!(du, u, t, source_terms::Nothing, equations::AbstractEquations{3}, cache)
    return nothing
end

# Pack kernels for calculating source terms 
function cuda_sources!(du, u, t, source_terms, equations::AbstractEquations{3}, cache)
    node_coordinates = cache.elements.node_coordinates

    source_terms_kernel = @cuda launch=false source_terms_kernel!(du, u, node_coordinates, t,
                                                                  equations, source_terms)
    source_terms_kernel(du, u, node_coordinates, t, equations, source_terms;
                        kernel_configurator_2d(source_terms_kernel, size(u, 2)^3, size(u, 5))...)

    return nothing
end

# Put everything together into a single function.

# See also `rhs!` function in Trixi.jl
function rhs_gpu!(du, u, t, mesh::TreeMesh{3}, equations, boundary_conditions,
                  source_terms::Source, dg::DGSEM, cache) where {Source}
    # reset_du!(du) 
    # reset_du!(du) is now fused into the next kernel, 
    # removing the need for a separate function call.

    cuda_volume_integral!(du, u, mesh, have_nonconservative_terms(equations), equations,
                          dg.volume_integral, dg, cache)

    cuda_prolong2interfaces!(u, mesh, equations, cache)

    cuda_interface_flux!(mesh, have_nonconservative_terms(equations), equations, dg, cache)

    cuda_prolong2boundaries!(u, mesh, boundary_conditions, equations, cache)

    cuda_boundary_flux!(t, mesh, boundary_conditions,
                        have_nonconservative_terms(equations), equations, dg, cache)

    cuda_prolong2mortars!(u, mesh, check_cache_mortars(cache), dg, cache)

    cuda_mortar_flux!(mesh, check_cache_mortars(cache), have_nonconservative_terms(equations),
                      equations, dg, cache)

    cuda_surface_integral!(du, mesh, equations, dg, cache)

    cuda_jacobian!(du, mesh, equations, cache)

    cuda_sources!(du, u, t, source_terms, equations, cache)

    return nothing
end
