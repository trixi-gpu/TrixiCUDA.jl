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

        @inbounds begin
            for ii in axes(u, 1)
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

        @inbounds begin
            for ii in axes(du, 2)
                du[i, j1, j2, j3, k] += derivative_dhat[j1, ii] * flux_arr1[i, ii, j2, j3, k]
                du[i, j1, j2, j3, k] += derivative_dhat[j2, ii] * flux_arr2[i, j1, ii, j3, k]
                du[i, j1, j2, j3, k] += derivative_dhat[j3, ii] * flux_arr3[i, j1, j2, ii, k]
            end
        end
    end

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

        @inbounds begin
            for ii in axes(u, 1)
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

        @inbounds begin
            for ii in axes(du, 2)
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

        orientation = orientations[k]
        left_element = neighbor_ids[1, k]
        right_element = neighbor_ids[2, k]

        @inbounds begin
            interfaces_u[1, j1, j2, j3, k] = u[j1,
                                               isequal(orientation, 1) * u2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j3,
                                               isequal(orientation, 1) * j3 + isequal(orientation, 2) * j3 + isequal(orientation, 3) * u2,
                                               left_element]
            interfaces_u[2, j1, j2, j3, k] = u[j1,
                                               isequal(orientation, 1) + isequal(orientation, 2) * j2 + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation,
                                                                                      2) + isequal(orientation, 3) * j3,
                                               isequal(orientation, 1) * j3 + isequal(orientation, 2) * j3 + isequal(orientation,
                                                                                                                     3),
                                               right_element]
        end
    end

    return nothing
end

# Kernel for calculating surface fluxes 
function surface_flux_kernel!(surface_flux_arr, interfaces_u, orientations,
                              equations::AbstractEquations{3}, surface_flux::Any)
    j2 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j3 = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (j2 <= size(surface_flux_arr, 3) &&
        j3 <= size(surface_flux_arr, 4) &&
        k <= size(surface_flux_arr, 5))
        u_ll, u_rr = get_surface_node_vars(interfaces_u, equations, j2, j3, k)
        orientation = orientations[k]

        surface_flux_node = surface_flux(u_ll, u_rr, orientation, equations)

        @inbounds begin
            for j1j1 in axes(surface_flux_arr, 2)
                surface_flux_arr[1, j1j1, j2, j3, k] = surface_flux_node[j1j1]
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

    if (i <= size(surface_flux_values, 1) &&
        j <= size(surface_flux_arr, 3)^2 &&
        k <= size(surface_flux_arr, 5))
        j1 = div(j - 1, size(surface_flux_arr, 3)) + 1
        j2 = rem(j - 1, size(surface_flux_arr, 3)) + 1

        left_id = neighbor_ids[1, k]
        right_id = neighbor_ids[2, k]

        left_dir = 2 * orientations[k]
        right_dir = 2 * orientations[k] - 1

        @inbounds begin
            surface_flux_values[i, j1, j2, left_dir, left_id] = surface_flux_arr[1, i, j1, j2, k]
            surface_flux_values[i, j1, j2, right_dir, right_id] = surface_flux_arr[1, i, j1, j2, k]
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

        element = neighbor_ids[k]
        side = neighbor_sides[k]
        orientation = orientations[k]

        @inbounds begin
            boundaries_u[1, j1, j2, j3, k] = u[j1,
                                               isequal(orientation, 1) * u2 + isequal(orientation, 2) * j2 + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * u2 + isequal(orientation, 3) * j3,
                                               isequal(orientation, 1) * j3 + isequal(orientation, 2) * j3 + isequal(orientation, 3) * u2,
                                               element] * (2 - side) # Set to 0 instead of NaN
            boundaries_u[2, j1, j2, j3, k] = u[j1,
                                               isequal(orientation, 1) + isequal(orientation, 2) * j2 + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation,
                                                                                      2) + isequal(orientation, 3) * j3,
                                               isequal(orientation, 1) * j3 + isequal(orientation, 2) * j3 + isequal(orientation,
                                                                                                                     3),
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

        boundary = boundary_arr[k]
        direction = (indices_arr[1] <= boundary) + (indices_arr[2] <= boundary) +
                    (indices_arr[3] <= boundary) + (indices_arr[4] <= boundary) +
                    (indices_arr[5] <= boundary) + (indices_arr[6] <= boundary)

        neighbor = neighbor_ids[boundary]
        side = neighbor_sides[boundary]
        orientation = orientations[boundary]

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

        @inbounds begin
            for ii in axes(surface_flux_values, 1)
                # `boundary_flux_node` can be nothing if periodic boundary condition is applied
                surface_flux_values[ii, j1, j2, direction, neighbor] = boundary_flux_node ===
                                                                       nothing ? # bad
                                                                       surface_flux_values[ii, j1,
                                                                                           j2,
                                                                                           direction,
                                                                                           neighbor] :
                                                                       boundary_flux_node[ii]
            end
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

        large_side = large_sides[k]
        orientation = orientations[k]

        lower_left_element = neighbor_ids[1, k]
        lower_right_element = neighbor_ids[2, k]
        upper_left_element = neighbor_ids[3, k]
        upper_right_element = neighbor_ids[4, k]

        @inbounds begin
            u_upper_left[2, i, j1, j2, k] = u[i,
                                              isequal(orientation, 1) + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1,
                                              isequal(orientation, 1) * j1 + isequal(orientation,
                                                                                     2) + isequal(orientation, 3) * j2,
                                              isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation,
                                                                                                                    3),
                                              upper_left_element] * (2 - large_side)

            u_upper_right[2, i, j1, j2, k] = u[i,
                                               isequal(orientation, 1) + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1,
                                               isequal(orientation, 1) * j1 + isequal(orientation,
                                                                                      2) + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation,
                                                                                                                     3),
                                               upper_right_element] * (2 - large_side)

            u_lower_left[2, i, j1, j2, k] = u[i,
                                              isequal(orientation, 1) + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1,
                                              isequal(orientation, 1) * j1 + isequal(orientation,
                                                                                     2) + isequal(orientation, 3) * j2,
                                              isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation,
                                                                                                                    3),
                                              lower_left_element] * (2 - large_side)

            u_lower_right[2, i, j1, j2, k] = u[i,
                                               isequal(orientation, 1) + isequal(orientation, 2) * j1 + isequal(orientation, 3) * j1,
                                               isequal(orientation, 1) * j1 + isequal(orientation,
                                                                                      2) + isequal(orientation, 3) * j2,
                                               isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation,
                                                                                                                     3),
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

# Kernel for interpolating data large to small on mortars
function prolong_mortars_large2small_kernel!(u_upper_left, u_upper_right, u_lower_left,
                                             u_lower_right, tmp_upper_left, tmp_upper_right,
                                             tmp_lower_left, tmp_lower_right, u, forward_upper,
                                             forward_lower, neighbor_ids, large_sides, orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u_upper_left, 2) && j <= size(u_upper_left, 3)^2 && k <= size(u_upper_left, 5))
        u2 = size(u, 2) # size(u_upper_left, 3) == size(u, 2)

        j1 = div(j - 1, u2) + 1
        j2 = rem(j - 1, u2) + 1

        large_side = large_sides[k]
        orientation = orientations[k]
        large_element = neighbor_ids[5, k]

        leftright = large_side

        @inbounds begin
            for j1j1 in axes(forward_lower, 2)
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

            for j1j1 in axes(forward_lower, 2)
                tmp_upper_left[leftright, i, j1, j2, k] += forward_lower[j1, j1j1] *
                                                           u[i,
                                                             isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
                                                             isequal(orientation, 1) * j1j1 + isequal(orientation,
                                                                                                      2) + isequal(orientation, 3) * j2,
                                                             isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation,
                                                                                                                                   3),
                                                             large_element] * (large_side - 1)

                tmp_upper_right[leftright, i, j1, j2, k] += forward_upper[j1, j1j1] *
                                                            u[i,
                                                              isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
                                                              isequal(orientation, 1) * j1j1 + isequal(orientation,
                                                                                                       2) + isequal(orientation, 3) * j2,
                                                              isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation,
                                                                                                                                    3),
                                                              large_element] * (large_side - 1)

                tmp_lower_left[leftright, i, j1, j2, k] += forward_lower[j1, j1j1] *
                                                           u[i,
                                                             isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
                                                             isequal(orientation, 1) * j1j1 + isequal(orientation,
                                                                                                      2) + isequal(orientation, 3) * j2,
                                                             isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation,
                                                                                                                                   3),
                                                             large_element] * (large_side - 1)

                tmp_lower_right[leftright, i, j1, j2, k] += forward_upper[j1, j1j1] *
                                                            u[i,
                                                              isequal(orientation, 1) + isequal(orientation, 2) * j1j1 + isequal(orientation, 3) * j1j1,
                                                              isequal(orientation, 1) * j1j1 + isequal(orientation,
                                                                                                       2) + isequal(orientation, 3) * j2,
                                                              isequal(orientation, 1) * j2 + isequal(orientation, 2) * j2 + isequal(orientation,
                                                                                                                                    3),
                                                              large_element] * (large_side - 1)
            end

            for j2j2 in axes(forward_upper, 2)
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
function mortar_flux_kernel!(fstar_upper_left, fstar_upper_right, fstar_lower_left,
                             fstar_lower_right, u_upper_left, u_upper_right, u_lower_left,
                             u_lower_right, orientations, equations, surface_flux::Any)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u_upper_left, 3) && j <= size(u_upper_left, 4) && k <= length(orientations))
        u_ll_upper_left, u_rr_upper_left = get_surface_node_vars(u_upper_left, equations, i, j, k)
        u_ll_upper_right, u_rr_upper_right = get_surface_node_vars(u_upper_right, equations, i, j,
                                                                   k)
        u_ll_lower_left, u_rr_lower_left = get_surface_node_vars(u_lower_left, equations, i, j, k)
        u_ll_lower_right, u_rr_lower_right = get_surface_node_vars(u_lower_right, equations, i, j,
                                                                   k)

        orientation = orientations[k]

        flux_upper_left_node = surface_flux(u_ll_upper_left, u_rr_upper_left, orientation,
                                            equations)
        flux_upper_right_node = surface_flux(u_ll_upper_right, u_rr_upper_right, orientation,
                                             equations)
        flux_lower_left_node = surface_flux(u_ll_lower_left, u_rr_lower_left, orientation,
                                            equations)
        flux_lower_right_node = surface_flux(u_ll_lower_right, u_rr_lower_right, orientation,
                                             equations)

        @inbounds begin
            for ii in axes(fstar_upper_left, 1)
                fstar_upper_left[ii, i, j, k] = flux_upper_left_node[ii]
                fstar_upper_right[ii, i, j, k] = flux_upper_right_node[ii]
                fstar_lower_left[ii, i, j, k] = flux_lower_left_node[ii]
                fstar_lower_right[ii, i, j, k] = flux_lower_right_node[ii]
            end
        end
    end

    return nothing
end

# Kernel for copying mortar fluxes small to small and small to large
function mortar_flux_copy_to_kernel!(surface_flux_values, tmp_surface_flux_values, tmp_upper_left,
                                     tmp_upper_right, tmp_lower_left, tmp_lower_right,
                                     fstar_upper_left,
                                     fstar_upper_right, fstar_lower_left, fstar_lower_right,
                                     reverse_upper,
                                     reverse_lower, neighbor_ids, large_sides, orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_values, 2)^2 &&
        k <= length(orientations))
        j1 = div(j - 1, size(surface_flux_values, 2)) + 1
        j2 = rem(j - 1, size(surface_flux_values, 2)) + 1

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

        surface_flux_values[i, j1, j2, direction, upper_left_element] = fstar_upper_left[i, j1, j2,
                                                                                         k]
        surface_flux_values[i, j1, j2, direction, upper_right_element] = fstar_upper_right[i, j1,
                                                                                           j2, k]
        surface_flux_values[i, j1, j2, direction, lower_left_element] = fstar_lower_left[i, j1, j2,
                                                                                         k]
        surface_flux_values[i, j1, j2, direction, lower_right_element] = fstar_lower_right[i, j1,
                                                                                           j2, k]

        # Use simple math expression to enhance the performance (against control flow), 
        # it is equivalent to, `isequal(large_side, 1) * isequal(orientation, 1) * 2 +
        #                       isequal(large_side, 1) * isequal(orientation, 2) * 4 +
        #                       isequal(large_side, 1) * isequal(orientation, 3) * 6 +
        #                       isequal(large_side, 2) * isequal(orientation, 1) * 1 +
        #                       isequal(large_side, 2) * isequal(orientation, 2) * 3 +
        #                       isequal(large_side, 2) * isequal(orientation, 3) * 5`.
        # Please also check the original code in Trixi.jl when you modify this code.
        direction = 2 * orientation - large_side + 1

        @inbounds begin
            for j1j1 in axes(reverse_upper, 2)
                tmp_upper_left[i, j1, j2, direction, large_element] += reverse_lower[j1, j1j1] *
                                                                       fstar_upper_left[i, j1j1, j2,
                                                                                        k]
                tmp_upper_right[i, j1, j2, direction, large_element] += reverse_upper[j1, j1j1] *
                                                                        fstar_upper_right[i, j1j1,
                                                                                          j2, k]
                tmp_lower_left[i, j1, j2, direction, large_element] += reverse_lower[j1, j1j1] *
                                                                       fstar_lower_left[i, j1j1, j2,
                                                                                        k]
                tmp_lower_right[i, j1, j2, direction, large_element] += reverse_upper[j1, j1j1] *
                                                                        fstar_lower_right[i, j1j1,
                                                                                          j2, k]
            end

            for j2j2 in axes(reverse_lower, 2)
                tmp_surface_flux_values[i, j1, j2, direction, large_element] += reverse_upper[j2,
                                                                                              j2j2] *
                                                                                tmp_upper_left[i,
                                                                                               j1,
                                                                                               j2j2,
                                                                                               direction,
                                                                                               large_element]
                tmp_surface_flux_values[i, j1, j2, direction, large_element] += reverse_upper[j2,
                                                                                              j2j2] *
                                                                                tmp_upper_right[i,
                                                                                                j1,
                                                                                                j2j2,
                                                                                                direction,
                                                                                                large_element]
                tmp_surface_flux_values[i, j1, j2, direction, large_element] += reverse_lower[j2,
                                                                                              j2j2] *
                                                                                tmp_lower_left[i,
                                                                                               j1,
                                                                                               j2j2,
                                                                                               direction,
                                                                                               large_element]
                tmp_surface_flux_values[i, j1, j2, direction, large_element] += reverse_lower[j2,
                                                                                              j2j2] *
                                                                                tmp_lower_right[i,
                                                                                                j1,
                                                                                                j2j2,
                                                                                                direction,
                                                                                                large_element]
            end

            surface_flux_values[i, j1, j2, direction, large_element] = tmp_surface_flux_values[i,
                                                                                               j1,
                                                                                               j2,
                                                                                               direction,
                                                                                               large_element]
        end
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
            du[i, j1, j2, j3, k] += (surface_flux_values[i, j2, j3, 2, k] *
                                     isequal(j1, u2) +
                                     surface_flux_values[i, j1, j3, 4, k] *
                                     isequal(j2, u2) +
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

        @inbounds begin
            for ii in axes(du, 1)
                du[ii, j1, j2, j3, k] += source_terms_node[ii]
            end
        end
    end

    return nothing
end

# Functions that begin with `cuda_` are the functions that pack CUDA kernels together to do 
# partial work in semidiscretization. They are used to invoke kernels from the host (i.e., CPU) 
# and run them on the device (i.e., GPU).

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::DGSEM)
    derivative_dhat = CuArray{Float64}(dg.basis.derivative_dhat)
    flux_arr1 = similar(u)
    flux_arr2 = similar(u)
    flux_arr3 = similar(u)

    size_arr = CuArray{Float64}(undef, size(u, 2)^3, size(u, 5))

    flux_kernel = @cuda launch=false flux_kernel!(flux_arr1, flux_arr2, flux_arr3, u, equations,
                                                  flux)
    flux_kernel(flux_arr1, flux_arr2, flux_arr3, u, equations, flux;
                configurator_2d(flux_kernel, size_arr)...)

    size_arr = CuArray{Float64}(undef, size(du, 1), size(du, 2)^3, size(du, 5))

    weak_form_kernel = @cuda launch=false weak_form_kernel!(du, derivative_dhat, flux_arr1,
                                                            flux_arr2, flux_arr3)
    weak_form_kernel(du, derivative_dhat, flux_arr1, flux_arr2, flux_arr3;
                     configurator_3d(weak_form_kernel, size_arr)...)

    return nothing
end

# Launch CUDA kernels to calculate volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM)
    volume_flux = volume_integral.volume_flux

    derivative_split = CuArray{Float64}(dg.basis.derivative_split)
    volume_flux_arr1 = CuArray{Float64}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                        size(u, 2), size(u, 5))
    volume_flux_arr2 = CuArray{Float64}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                        size(u, 2), size(u, 5))
    volume_flux_arr3 = CuArray{Float64}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                        size(u, 2), size(u, 5))

    size_arr = CuArray{Float64}(undef, size(u, 2)^4, size(u, 5))

    volume_flux_kernel = @cuda launch=false volume_flux_kernel!(volume_flux_arr1, volume_flux_arr2,
                                                                volume_flux_arr3, u, equations,
                                                                volume_flux)
    volume_flux_kernel(volume_flux_arr1, volume_flux_arr2, volume_flux_arr3, u, equations,
                       volume_flux;
                       configurator_2d(volume_flux_kernel, size_arr)...)

    size_arr = CuArray{Float64}(undef, size(du, 1), size(du, 2)^3, size(du, 5))

    volume_integral_kernel = @cuda launch=false volume_integral_kernel!(du, derivative_split,
                                                                        volume_flux_arr1,
                                                                        volume_flux_arr2,
                                                                        volume_flux_arr3, equations)
    volume_integral_kernel(du, derivative_split, volume_flux_arr1, volume_flux_arr2,
                           volume_flux_arr3, equations;
                           configurator_3d(volume_integral_kernel, size_arr)...)

    return nothing
end

function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms::True, equations,
                               volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM)
    # Wait for the unmutable MHD implementation in Trixi.jl
end

# Pack kernels to prolonging solution to interfaces
function cuda_prolong2interfaces!(u, mesh::TreeMesh{3}, equations, cache)
    neighbor_ids = CuArray{Int64}(cache.interfaces.neighbor_ids)
    orientations = CuArray{Int64}(cache.interfaces.orientations)
    interfaces_u = CuArray{Float64}(cache.interfaces.u)

    size_arr = CuArray{Float64}(undef, size(interfaces_u, 2) * size(interfaces_u, 3)^2,
                                size(interfaces_u, 5))

    prolong_interfaces_kernel = @cuda launch=false prolong_interfaces_kernel!(interfaces_u, u,
                                                                              neighbor_ids,
                                                                              orientations,
                                                                              equations)
    prolong_interfaces_kernel(interfaces_u, u, neighbor_ids, orientations, equations;
                              configurator_2d(prolong_interfaces_kernel, size_arr)...)

    cache.interfaces.u = interfaces_u  # copy back to host automatically

    return nothing
end

# Pack kernels for calculating interface fluxes
function cuda_interface_flux!(mesh::TreeMesh{3}, nonconservative_terms::False, equations, dg::DGSEM,
                              cache)
    surface_flux = dg.surface_integral.surface_flux

    neighbor_ids = CuArray{Int64}(cache.interfaces.neighbor_ids)
    orientations = CuArray{Int64}(cache.interfaces.orientations)
    interfaces_u = CuArray{Float64}(cache.interfaces.u)
    surface_flux_arr = CuArray{Float64}(undef, 1, size(interfaces_u)[2:end]...)
    surface_flux_values = CuArray{Float64}(cache.elements.surface_flux_values)

    size_arr = CuArray{Float64}(undef, size(interfaces_u, 3), size(interfaces_u, 4),
                                size(interfaces_u, 5))

    surface_flux_kernel = @cuda launch=false surface_flux_kernel!(surface_flux_arr, interfaces_u,
                                                                  orientations, equations,
                                                                  surface_flux)
    surface_flux_kernel(surface_flux_arr, interfaces_u, orientations, equations, surface_flux;
                        configurator_3d(surface_flux_kernel, size_arr)...)

    size_arr = CuArray{Float64}(undef, size(surface_flux_values, 1), size(interfaces_u, 3)^2,
                                size(interfaces_u, 5))

    interface_flux_kernel = @cuda launch=false interface_flux_kernel!(surface_flux_values,
                                                                      surface_flux_arr,
                                                                      neighbor_ids, orientations,
                                                                      equations)
    interface_flux_kernel(surface_flux_values, surface_flux_arr, neighbor_ids, orientations,
                          equations;
                          configurator_3d(interface_flux_kernel, size_arr)...)

    cache.elements.surface_flux_values = surface_flux_values # copy back to host automatically

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
    neighbor_ids = CuArray{Int64}(cache.boundaries.neighbor_ids)
    neighbor_sides = CuArray{Int64}(cache.boundaries.neighbor_sides)
    orientations = CuArray{Int64}(cache.boundaries.orientations)
    boundaries_u = CuArray{Float64}(cache.boundaries.u)

    size_arr = CuArray{Float64}(undef, size(boundaries_u, 2) * size(boundaries_u, 3)^2,
                                size(boundaries_u, 5))

    prolong_boundaries_kernel = @cuda launch=false prolong_boundaries_kernel!(boundaries_u, u,
                                                                              neighbor_ids,
                                                                              neighbor_sides,
                                                                              orientations,
                                                                              equations)
    prolong_boundaries_kernel(boundaries_u, u, neighbor_ids, neighbor_sides, orientations,
                              equations;
                              configurator_2d(prolong_boundaries_kernel, size_arr)...)

    cache.boundaries.u = boundaries_u  # copy back to host automatically

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

    n_boundaries_per_direction = CuArray{Int64}(cache.boundaries.n_boundaries_per_direction)
    neighbor_ids = CuArray{Int64}(cache.boundaries.neighbor_ids)
    neighbor_sides = CuArray{Int64}(cache.boundaries.neighbor_sides)
    orientations = CuArray{Int64}(cache.boundaries.orientations)
    boundaries_u = CuArray{Float64}(cache.boundaries.u)
    node_coordinates = CuArray{Float64}(cache.boundaries.node_coordinates)
    surface_flux_values = CuArray{Float64}(cache.elements.surface_flux_values)

    lasts = zero(n_boundaries_per_direction)
    firsts = zero(n_boundaries_per_direction)

    last_first_indices_kernel = @cuda launch=false last_first_indices_kernel!(lasts, firsts,
                                                                              n_boundaries_per_direction)
    last_first_indices_kernel(lasts, firsts, n_boundaries_per_direction;
                              configurator_1d(last_first_indices_kernel, lasts)...)

    lasts, firsts = Array(lasts), Array(firsts)
    boundary_arr = CuArray{Int64}(firsts[1]:lasts[6])
    indices_arr = CuArray{Int64}([firsts[1], firsts[2], firsts[3], firsts[4], firsts[5], firsts[6]])
    boundary_conditions_callable = replace_boundary_conditions(boundary_conditions)

    size_arr = CuArray{Float64}(undef, size(surface_flux_values, 2)^2, length(boundary_arr))

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
                         configurator_2d(boundary_flux_kernel, size_arr)...)

    cache.elements.surface_flux_values = surface_flux_values # copy back to host automatically

    return nothing
end

# Dummy function for asserting mortars 
function cuda_prolong2mortars!(u, mesh::TreeMesh{3}, cache_mortars::False, dg::DGSEM, cache)
    @assert iszero(length(cache.mortars.orientations))
end

# Pack kernels for prolonging solution to mortars
function cuda_prolong2mortars!(u, mesh::TreeMesh{3}, cache_mortars::True, dg::DGSEM, cache)
    neighbor_ids = CuArray{Int64}(cache.mortars.neighbor_ids)
    large_sides = CuArray{Int64}(cache.mortars.large_sides)
    orientations = CuArray{Int64}(cache.mortars.orientations)

    u_upper_left = zero(CuArray{Float64}(cache.mortars.u_upper_left)) # NaN to zero
    u_upper_right = zero(CuArray{Float64}(cache.mortars.u_upper_right)) # NaN to zero
    u_lower_left = zero(CuArray{Float64}(cache.mortars.u_lower_left)) # NaN to zero
    u_lower_right = zero(CuArray{Float64}(cache.mortars.u_lower_right)) # NaN to zero

    forward_upper = CuArray{Float64}(dg.mortar.forward_upper)
    forward_lower = CuArray{Float64}(dg.mortar.forward_lower)

    size_arr = CuArray{Float64}(undef, size(u_upper_left, 2), size(u_upper_left, 3)^2,
                                size(u_upper_left, 5))

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
                                       configurator_3d(prolong_mortars_small2small_kernel,
                                                       size_arr)...)
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
                                                                                                u,
                                                                                                forward_upper,
                                                                                                forward_lower,
                                                                                                neighbor_ids,
                                                                                                large_sides,
                                                                                                orientations)
    prolong_mortars_large2small_kernel(u_upper_left, u_upper_right, u_lower_left, u_lower_right,
                                       tmp_upper_left, tmp_upper_right, tmp_lower_left,
                                       tmp_lower_right, u,
                                       forward_upper, forward_lower, neighbor_ids, large_sides,
                                       orientations;
                                       configurator_3d(prolong_mortars_large2small_kernel,
                                                       size_arr)...)

    cache.mortars.u_upper_left = u_upper_left # copy back to host automatically
    cache.mortars.u_upper_right = u_upper_right # copy back to host automatically
    cache.mortars.u_lower_left = u_lower_left # copy back to host automatically
    cache.mortars.u_lower_right = u_lower_right # copy back to host automatically

    return nothing
end

# Dummy function for asserting mortar fluxes
function cuda_mortar_flux!(mesh::TreeMesh{3}, cache_mortars::False, nonconservative_terms,
                           equations, dg::DGSEM, cache)
    @assert iszero(length(cache.mortars.orientations))
end

# Pack kernels for calculating mortar fluxes
function cuda_mortar_flux!(mesh::TreeMesh{3}, cache_mortars::True, nonconservative_terms::False,
                           equations, dg::DGSEM, cache)
    surface_flux = dg.surface_integral.surface_flux

    neighbor_ids = CuArray{Int64}(cache.mortars.neighbor_ids)
    large_sides = CuArray{Int64}(cache.mortars.large_sides)
    orientations = CuArray{Int64}(cache.mortars.orientations)

    u_upper_left = CuArray{Float64}(cache.mortars.u_upper_left)
    u_upper_right = CuArray{Float64}(cache.mortars.u_upper_right)
    u_lower_left = CuArray{Float64}(cache.mortars.u_lower_left)
    u_lower_right = CuArray{Float64}(cache.mortars.u_lower_right)
    reverse_upper = CuArray{Float64}(dg.mortar.reverse_upper)
    reverse_lower = CuArray{Float64}(dg.mortar.reverse_lower)

    surface_flux_values = CuArray{Float64}(cache.elements.surface_flux_values)
    tmp_surface_flux_values = zero(similar(surface_flux_values)) # undef to zero

    fstar_upper_left = CuArray{Float64}(undef, size(u_upper_left, 2), size(u_upper_left, 3),
                                        size(u_upper_left, 4), length(orientations))
    fstar_upper_right = CuArray{Float64}(undef, size(u_upper_left, 2), size(u_upper_left, 3),
                                         size(u_upper_left, 4), length(orientations))
    fstar_lower_left = CuArray{Float64}(undef, size(u_upper_left, 2), size(u_upper_left, 3),
                                        size(u_upper_left, 4), length(orientations))
    fstar_lower_right = CuArray{Float64}(undef, size(u_upper_left, 2), size(u_upper_left, 3),
                                         size(u_upper_left, 4), length(orientations))

    size_arr = CuArray{Float64}(undef, size(u_upper_left, 3), size(u_upper_left, 4),
                                length(orientations))

    mortar_flux_kernel = @cuda launch=false mortar_flux_kernel!(fstar_upper_left, fstar_upper_right,
                                                                fstar_lower_left, fstar_lower_right,
                                                                u_upper_left, u_upper_right,
                                                                u_lower_left, u_lower_right,
                                                                orientations, equations,
                                                                surface_flux)
    mortar_flux_kernel(fstar_upper_left, fstar_upper_right, fstar_lower_left, fstar_lower_right,
                       u_upper_left, u_upper_right, u_lower_left, u_lower_right, orientations,
                       equations, surface_flux;
                       configurator_3d(mortar_flux_kernel, size_arr)...)

    tmp_upper_left = zero(similar(surface_flux_values)) # undef to zero
    tmp_upper_right = zero(similar(surface_flux_values)) # undef to zero
    tmp_lower_left = zero(similar(surface_flux_values)) # undef to zero
    tmp_lower_right = zero(similar(surface_flux_values)) # undef to zero

    size_arr = CuArray{Float64}(undef, size(surface_flux_values, 1), size(surface_flux_values, 2)^2,
                                length(orientations))

    mortar_flux_copy_to_kernel = @cuda launch=false mortar_flux_copy_to_kernel!(surface_flux_values,
                                                                                tmp_surface_flux_values,
                                                                                tmp_upper_left,
                                                                                tmp_upper_right,
                                                                                tmp_lower_left,
                                                                                tmp_lower_right,
                                                                                fstar_upper_left,
                                                                                fstar_upper_right,
                                                                                fstar_lower_left,
                                                                                fstar_lower_right,
                                                                                reverse_upper,
                                                                                reverse_lower,
                                                                                neighbor_ids,
                                                                                large_sides,
                                                                                orientations)
    mortar_flux_copy_to_kernel(surface_flux_values, tmp_surface_flux_values, tmp_upper_left,
                               tmp_upper_right, tmp_lower_left, tmp_lower_right, fstar_upper_left,
                               fstar_upper_right, fstar_lower_left, fstar_lower_right,
                               reverse_upper,
                               reverse_lower, neighbor_ids, large_sides, orientations;
                               configurator_3d(mortar_flux_copy_to_kernel, size_arr)...)

    cache.elements.surface_flux_values = surface_flux_values # copy back to host automatically
end

function cuda_mortar_flux!(mesh::TreeMesh{3}, cache_mortars::True, nonconservative_terms::True,
                           equations, dg::DGSEM, cache)
end

# Pack kernels for calculating surface integrals
function cuda_surface_integral!(du, mesh::TreeMesh{3}, equations, dg::DGSEM, cache)
    factor_arr = CuArray{Float64}([
                                      dg.basis.boundary_interpolation[1, 1],
                                      dg.basis.boundary_interpolation[size(du, 2), 2]
                                  ])
    surface_flux_values = CuArray{Float64}(cache.elements.surface_flux_values)

    size_arr = CuArray{Float64}(undef, size(du, 1), size(du, 2)^3, size(du, 5))

    surface_integral_kernel = @cuda launch=false surface_integral_kernel!(du, factor_arr,
                                                                          surface_flux_values,
                                                                          equations)
    surface_integral_kernel(du, factor_arr, surface_flux_values, equations;
                            configurator_3d(surface_integral_kernel, size_arr)...)

    return nothing
end

# Pack kernels for applying Jacobian to reference element
function cuda_jacobian!(du, mesh::TreeMesh{3}, equations, cache)
    inverse_jacobian = CuArray{Float64}(cache.elements.inverse_jacobian)

    size_arr = CuArray{Float64}(undef, size(du, 1), size(du, 2)^3, size(du, 5))

    jacobian_kernel = @cuda launch=false jacobian_kernel!(du, inverse_jacobian, equations)
    jacobian_kernel(du, inverse_jacobian, equations; configurator_3d(jacobian_kernel, size_arr)...)

    return nothing
end

# Dummy function returning nothing            
function cuda_sources!(du, u, t, source_terms::Nothing, equations::AbstractEquations{3}, cache)
    return nothing
end

# Pack kernels for calculating source terms 
function cuda_sources!(du, u, t, source_terms, equations::AbstractEquations{3}, cache)
    node_coordinates = CuArray{Float64}(cache.elements.node_coordinates)

    size_arr = CuArray{Float64}(undef, size(u, 2)^3, size(u, 5))

    source_terms_kernel = @cuda launch=false source_terms_kernel!(du, u, node_coordinates, t,
                                                                  equations, source_terms)
    source_terms_kernel(du, u, node_coordinates, t, equations, source_terms;
                        configurator_2d(source_terms_kernel, size_arr)...)

    return nothing
end

# Put everything together into a single function.

# Ref: `rhs!` function in Trixi.jl
function rhs_gpu!(du_cpu, u_cpu, t, mesh::TreeMesh{3}, equations, boundary_conditions,
                  source_terms::Source, dg::DGSEM, cache) where {Source}
    du, u = copy_to_device!(du_cpu, u_cpu)

    cuda_volume_integral!(du, u, mesh, have_nonconservative_terms(equations), equations,
                          dg.volume_integral, dg)

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

    du_computed, _ = copy_to_host!(du, u)
    du_cpu .= du_computed

    return nothing
end
