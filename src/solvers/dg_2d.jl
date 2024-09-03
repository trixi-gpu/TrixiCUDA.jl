# Everything related to a DG semidiscretization in 2D.

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

        @inbounds begin
            for ii in axes(u, 1)
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

        @inbounds begin
            for ii in axes(du, 2)
                du[i, j1, j2, k] += derivative_dhat[j1, ii] * flux_arr1[i, ii, j2, k]
                du[i, j1, j2, k] += derivative_dhat[j2, ii] * flux_arr2[i, j1, ii, k]
            end
        end
    end

    return nothing
end

# Kernel for calculating volume fluxes
function volume_flux_kernel!(volume_flux_arr1, volume_flux_arr2, u, equations::AbstractEquations{2},
                             volume_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^3 && k <= size(u, 4))
        j1 = div(j - 1, size(u, 2)^2) + 1
        j2 = div(rem(j - 1, size(u, 2)^2), size(u, 2)) + 1
        j3 = rem(rem(j - 1, size(u, 2)^2), size(u, 2)) + 1

        u_node = get_node_vars(u, equations, j1, j2, k)
        u_node1 = get_node_vars(u, equations, j3, j2, k)
        u_node2 = get_node_vars(u, equations, j1, j3, k)

        volume_flux_node1 = volume_flux(u_node, u_node1, 1, equations)
        volume_flux_node2 = volume_flux(u_node, u_node2, 2, equations)

        @inbounds begin
            for ii in axes(u, 1)
                volume_flux_arr1[ii, j1, j3, j2, k] = volume_flux_node1[ii]
                volume_flux_arr2[ii, j1, j2, j3, k] = volume_flux_node2[ii]
            end
        end
    end

    return nothing
end

# Kernel for calculating symmetric and nonconservative fluxes
function symmetric_noncons_flux_kernel!(symmetric_flux_arr1, symmetric_flux_arr2, noncons_flux_arr1,
                                        noncons_flux_arr2, u, derivative_split,
                                        equations::AbstractEquations{2}, symmetric_flux::Any,
                                        nonconservative_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^3 && k <= size(u, 4))
        j1 = div(j - 1, size(u, 2)^2) + 1
        j2 = div(rem(j - 1, size(u, 2)^2), size(u, 2)) + 1
        j3 = rem(rem(j - 1, size(u, 2)^2), size(u, 2)) + 1

        u_node = get_node_vars(u, equations, j1, j2, k)
        u_node1 = get_node_vars(u, equations, j3, j2, k)
        u_node2 = get_node_vars(u, equations, j1, j3, k)

        symmetric_flux_node1 = symmetric_flux(u_node, u_node1, 1, equations)
        symmetric_flux_node2 = symmetric_flux(u_node, u_node2, 2, equations)

        noncons_flux_node1 = nonconservative_flux(u_node, u_node1, 1, equations)
        noncons_flux_node2 = nonconservative_flux(u_node, u_node2, 2, equations)

        @inbounds begin
            for ii in axes(u, 1)
                symmetric_flux_arr1[ii, j1, j3, j2, k] = derivative_split[j1, j3] *
                                                         symmetric_flux_node1[ii]
                symmetric_flux_arr2[ii, j1, j2, j3, k] = derivative_split[j2, j3] *
                                                         symmetric_flux_node2[ii]
                noncons_flux_arr1[ii, j1, j3, j2, k] = noncons_flux_node1[ii]
                noncons_flux_arr2[ii, j1, j2, j3, k] = noncons_flux_node2[ii]
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

        @inbounds begin
            for ii in axes(du, 2)
                du[i, j1, j2, k] += derivative_split[j1, ii] * volume_flux_arr1[i, j1, ii, j2, k]
                du[i, j1, j2, k] += derivative_split[j2, ii] * volume_flux_arr2[i, j1, j2, ii, k]
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

        @inbounds begin
            integral_contribution = 0.0 # change back to `Float32` 

            for ii in axes(du, 2)
                du[i, j1, j2, k] += symmetric_flux_arr1[i, j1, ii, j2, k]
                du[i, j1, j2, k] += symmetric_flux_arr2[i, j1, j2, ii, k]
                integral_contribution += derivative_split[j1, ii] *
                                         noncons_flux_arr1[i, j1, ii, j2, k]
                integral_contribution += derivative_split[j2, ii] *
                                         noncons_flux_arr2[i, j1, j2, ii, k]
            end

            du[i, j1, j2, k] += 0.5f0 * integral_contribution # change back to `Float32`
        end
    end

    return nothing
end

# Kernel for prolonging two interfaces 
function prolong_interfaces_kernel!(interfaces_u, u, neighbor_ids, orientations,
                                    euqations::AbstractEquations{2})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(interfaces_u, 2) * size(interfaces_u, 3) && k <= size(interfaces_u, 4))
        j1 = div(j - 1, size(interfaces_u, 3)) + 1
        j2 = rem(j - 1, size(interfaces_u, 3)) + 1

        orientation = orientations[k]
        left_element = neighbor_ids[1, k]
        right_element = neighbor_ids[2, k]

        u2 = size(u, 2)

        @inbounds begin
            interfaces_u[1, j1, j2, k] = u[j1,
                                           isequal(orientation, 1) * u2 + isequal(orientation, 2) * j2,
                                           isequal(orientation, 1) * j2 + isequal(orientation, 2) * u2,
                                           left_element]
            interfaces_u[2, j1, j2, k] = u[j1,
                                           isequal(orientation, 1) * 1 + isequal(orientation, 2) * j2,
                                           isequal(orientation, 1) * j2 + isequal(orientation, 2) * 1,
                                           right_element]
        end
    end

    return nothing
end

# Kernel for calculating surface fluxes 
function surface_flux_kernel!(surface_flux_arr, interfaces_u, orientations,
                              equations::AbstractEquations{2}, surface_flux::Any)
    j2 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j2 <= size(surface_flux_arr, 3) && k <= size(surface_flux_arr, 4))
        u_ll, u_rr = get_surface_node_vars(interfaces_u, equations, j2, k)
        orientation = orientations[k]

        surface_flux_node = surface_flux(u_ll, u_rr, orientation, equations)

        @inbounds begin
            for j1j1 in axes(surface_flux_arr, 2)
                surface_flux_arr[1, j1j1, j2, k] = surface_flux_node[j1j1]
            end
        end
    end

    return nothing
end

# Kernel for calculating surface and both nonconservative fluxes 
function surface_noncons_flux_kernel!(surface_flux_arr, interfaces_u, noncons_left_arr,
                                      noncons_right_arr, orientations,
                                      equations::AbstractEquations{2}, surface_flux::Any,
                                      nonconservative_flux::Any)
    j2 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j2 <= size(surface_flux_arr, 3) && k <= size(surface_flux_arr, 4))
        u_ll, u_rr = get_surface_node_vars(interfaces_u, equations, j2, k)
        orientation = orientations[k]

        surface_flux_node = surface_flux(u_ll, u_rr, orientation, equations)
        noncons_left_node = nonconservative_flux(u_ll, u_rr, orientation, equations)
        noncons_right_node = nonconservative_flux(u_rr, u_ll, orientation, equations)

        @inbounds begin
            for j1j1 in axes(surface_flux_arr, 2)
                surface_flux_arr[1, j1j1, j2, k] = surface_flux_node[j1j1]
                noncons_left_arr[1, j1j1, j2, k] = noncons_left_node[j1j1]
                noncons_right_arr[1, j1j1, j2, k] = noncons_right_node[j1j1]
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

    if (i <= size(surface_flux_values, 1) &&
        j <= size(surface_flux_arr, 3) &&
        k <= size(surface_flux_arr, 4))
        left_id = neighbor_ids[1, k]
        right_id = neighbor_ids[2, k]

        left_direction = 2 * orientations[k]
        right_direction = 2 * orientations[k] - 1

        @inbounds begin
            surface_flux_values[i, j, left_direction, left_id] = surface_flux_arr[1, i, j, k]
            surface_flux_values[i, j, right_direction, right_id] = surface_flux_arr[1, i, j, k]
        end
    end

    return nothing
end

function interface_flux_kernel!(surface_flux_values, surface_flux_arr, noncons_left_arr,
                                noncons_right_arr, neighbor_ids, orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) &&
        j <= size(surface_flux_arr, 3) &&
        k <= size(surface_flux_arr, 4))
        left_id = neighbor_ids[1, k]
        right_id = neighbor_ids[2, k]

        left_direction = 2 * orientations[k]
        right_direction = 2 * orientations[k] - 1

        @inbounds begin
            surface_flux_values[i, j, left_direction, left_id] = surface_flux_arr[1, i, j, k] +
                                                                 0.5 * # change back to `Float32`
                                                                 noncons_left_arr[1, i, j, k]
            surface_flux_values[i, j, right_direction, right_id] = surface_flux_arr[1, i, j, k] +
                                                                   0.5 * # change back to `Float32`
                                                                   noncons_right_arr[1, i, j, k]
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
        j1 = div(j - 1, size(boundaries_u, 3)) + 1
        j2 = rem(j - 1, size(boundaries_u, 3)) + 1

        element = neighbor_ids[k]
        side = neighbor_sides[k]
        orientation = orientations[k]

        u2 = size(u, 2)

        @inbounds begin
            boundaries_u[1, j1, j2, k] = u[j1,
                                           isequal(orientation, 1) * u2 + isequal(orientation, 2) * j2,
                                           isequal(orientation, 1) * j2 + isequal(orientation, 2) * u2,
                                           element] * isequal(side, 1) # Set to 0 instead of NaN
            boundaries_u[2, j1, j2, k] = u[j1,
                                           isequal(orientation, 1) * 1 + isequal(orientation, 2) * j2,
                                           isequal(orientation, 1) * j2 + isequal(orientation, 2) * 1,
                                           element] * (1 - isequal(side, 1)) # Set to 0 instead of NaN
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
        boundary = boundary_arr[k]
        direction = (indices_arr[1] <= boundary) + (indices_arr[2] <= boundary) +
                    (indices_arr[3] <= boundary) + (indices_arr[4] <= boundary)

        neighbor = neighbor_ids[boundary]
        side = neighbor_sides[boundary]
        orientation = orientations[boundary]

        u_ll, u_rr = get_surface_node_vars(boundaries_u, equations, j, boundary)
        u_inner = isequal(side, 1) * u_ll + (1 - isequal(side, 1)) * u_rr
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

        @inbounds begin
            for ii in axes(surface_flux_values, 1)
                surface_flux_values[ii, j, direction, neighbor] = boundary_flux_node === nothing ? # bad
                                                                  surface_flux_values[ii, j,
                                                                                      direction,
                                                                                      neighbor] :
                                                                  boundary_flux_node[ii]
            end
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
        large_side = large_sides[k]
        orientation = orientations[k]

        lower_element = neighbor_ids[1, k]
        upper_element = neighbor_ids[2, k]

        u2 = size(u, 2)

        @inbounds begin
            u_upper[2, i, j, k] = u[i,
                                    isequal(orientation, 1) * 1 + isequal(orientation, 2) * j,
                                    isequal(orientation, 1) * j + isequal(orientation, 2) * 1,
                                    upper_element] * isequal(large_side, 1)

            u_lower[2, i, j, k] = u[i,
                                    isequal(orientation, 1) * 1 + isequal(orientation, 2) * j,
                                    isequal(orientation, 1) * j + isequal(orientation, 2) * 1,
                                    lower_element] * isequal(large_side, 1)

            u_upper[1, i, j, k] = u[i,
                                    isequal(orientation, 1) * u2 + isequal(orientation, 2) * j,
                                    isequal(orientation, 1) * j + isequal(orientation, 2) * u2,
                                    upper_element] * isequal(large_side, 2)

            u_lower[1, i, j, k] = u[i,
                                    isequal(orientation, 1) * u2 + isequal(orientation, 2) * j,
                                    isequal(orientation, 1) * j + isequal(orientation, 2) * u2,
                                    lower_element] * isequal(large_side, 2)
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
        large_side = large_sides[k]
        orientation = orientations[k]
        large_element = neighbor_ids[3, k]

        leftright = large_side
        u2 = size(u, 2)

        @inbounds begin
            for jj in axes(forward_upper, 2)
                u_upper[leftright, i, j, k] += forward_upper[j, jj] *
                                               u[i,
                                                 isequal(orientation, 1) * u2 + isequal(orientation, 2) * jj,
                                                 isequal(orientation, 1) * jj + isequal(orientation, 2) * u2,
                                                 large_element] * isequal(large_side, 1)
                u_lower[leftright, i, j, k] += forward_lower[j, jj] *
                                               u[i,
                                                 isequal(orientation, 1) * u2 + isequal(orientation, 2) * jj,
                                                 isequal(orientation, 1) * jj + isequal(orientation, 2) * u2,
                                                 large_element] * isequal(large_side, 1)
            end

            for jj in axes(forward_lower, 2)
                u_upper[leftright, i, j, k] += forward_upper[j, jj] *
                                               u[i,
                                                 isequal(orientation, 1) * 1 + isequal(orientation, 2) * jj,
                                                 isequal(orientation, 1) * jj + isequal(orientation, 2) * 1,
                                                 large_element] * isequal(large_side, 2)
                u_lower[leftright, i, j, k] += forward_lower[j, jj] *
                                               u[i,
                                                 isequal(orientation, 1) * 1 + isequal(orientation, 2) * jj,
                                                 isequal(orientation, 1) * jj + isequal(orientation, 2) * 1,
                                                 large_element] * isequal(large_side, 2)
            end
        end
    end

    return nothing
end

# Kernel for calculating mortar fluxes
function mortar_flux_kernel!(fstar_upper, fstar_lower, u_upper, u_lower, orientations, equations,
                             surface_flux::Any)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u_upper, 3) && k <= length(orientations))
        u_ll_upper, u_rr_upper = get_surface_node_vars(u_upper, equations, j, k)
        u_ll_lower, u_rr_lower = get_surface_node_vars(u_lower, equations, j, k)
        orientation = orientations[k]

        flux_upper_node = surface_flux(u_ll_upper, u_rr_upper, orientation, equations)
        flux_lower_node = surface_flux(u_ll_lower, u_rr_lower, orientation, equations)

        @inbounds begin
            for i in axes(fstar_upper, 1)
                fstar_upper[i, j, k] = flux_upper_node[i]
                fstar_lower[i, j, k] = flux_lower_node[i]
            end
        end
    end

    return nothing
end

# Kernel for copying mortar fluxes small to small and small to large
function mortar_flux_copy_to_kernel!(surface_flux_values, tmp_surface_flux_values, fstar_upper,
                                     fstar_lower, reverse_upper, reverse_lower, neighbor_ids,
                                     large_sides, orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_values, 2) &&
        k <= length(orientations))
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

        surface_flux_values[i, j, direction, upper_element] = fstar_upper[i, j, k]
        surface_flux_values[i, j, direction, lower_element] = fstar_lower[i, j, k]

        # Use math expression to enhance performance (against control flow), it is equivalent to,
        # `(2 - large_side) * (2 - orientation) * 2 + 
        #  (2 - large_side) * (orientation - 1) * 4 +
        #  (large_side - 1) * (2 - orientation) * 1 +
        #  (large_side - 1) * (orientation - 1) * 3`.
        direction = 2 * orientation - large_side + 1

        @inbounds begin
            for ii in axes(reverse_upper, 2) # i.e., ` for ii in axes(reverse_lower, 2)`
                tmp_surface_flux_values[i, j, direction, large_element] += reverse_upper[j, ii] *
                                                                           fstar_upper[i, ii, k] +
                                                                           reverse_lower[j, ii] *
                                                                           fstar_lower[i, ii, k]
            end

            surface_flux_values[i, j, direction, large_element] = tmp_surface_flux_values[i, j,
                                                                                          direction,
                                                                                          large_element]
        end
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
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds begin
            du[i, j1, j2, k] -= (surface_flux_values[i, j2, 1, k] * isequal(j1, 1) +
                                 surface_flux_values[i, j1, 3, k] * isequal(j2, 1)) * factor_arr[1]
            du[i, j1, j2, k] += (surface_flux_values[i, j2, 2, k] * isequal(j1, size(du, 2)) +
                                 surface_flux_values[i, j1, 4, k] * isequal(j2, size(du, 2))) *
                                factor_arr[2]
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

        @inbounds begin
            for ii in axes(du, 1)
                du[ii, j1, j2, k] += source_terms_node[ii]
            end
        end
    end

    return nothing
end

# Functions that begin with `cuda_` are the functions that pack CUDA kernels together to do 
# partial work in semidiscretization. They are used to invoke kernels from the host (i.e., CPU) 
# and run them on the device (i.e., GPU).

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{2}, nonconservative_terms, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::DGSEM)
    derivative_dhat = CuArray{Float64}(dg.basis.derivative_dhat)
    flux_arr1 = similar(u)
    flux_arr2 = similar(u)

    size_arr = CuArray{Float64}(undef, size(u, 2)^2, size(u, 4))

    flux_kernel = @cuda launch=false flux_kernel!(flux_arr1, flux_arr2, u, equations, flux)
    flux_kernel(flux_arr1, flux_arr2, u, equations, flux; configurator_2d(flux_kernel, size_arr)...)

    size_arr = CuArray{Float64}(undef, size(du, 1), size(du, 2)^2, size(du, 4))

    weak_form_kernel = @cuda launch=false weak_form_kernel!(du, derivative_dhat, flux_arr1,
                                                            flux_arr2)
    weak_form_kernel(du, derivative_dhat, flux_arr1, flux_arr2;
                     configurator_3d(weak_form_kernel, size_arr)...)

    return nothing
end

function cuda_volume_integral!(du, u, mesh::TreeMesh{2}, nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM)
    volume_flux = volume_integral.volume_flux

    derivative_split = dg.basis.derivative_split
    set_diagonal_to_zero!(derivative_split) # temporarily set here, maybe move outside `rhs!`

    derivative_split = CuArray{Float64}(derivative_split)
    volume_flux_arr1 = CuArray{Float64}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                        size(u, 4))
    volume_flux_arr2 = CuArray{Float64}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                        size(u, 4))

    size_arr = CuArray{Float64}(undef, size(u, 2)^3, size(u, 4))

    volume_flux_kernel = @cuda launch=false volume_flux_kernel!(volume_flux_arr1, volume_flux_arr2,
                                                                u, equations, volume_flux)
    volume_flux_kernel(volume_flux_arr1, volume_flux_arr2, u, equations, volume_flux;
                       configurator_2d(volume_flux_kernel, size_arr)...)

    size_arr = CuArray{Float64}(undef, size(du, 1), size(du, 2)^2, size(du, 4))

    volume_integral_kernel = @cuda launch=false volume_integral_kernel!(du, derivative_split,
                                                                        volume_flux_arr1,
                                                                        volume_flux_arr2, equations)
    volume_integral_kernel(du, derivative_split, volume_flux_arr1, volume_flux_arr2, equations;
                           configurator_3d(volume_integral_kernel, size_arr)...)

    return nothing
end

function cuda_volume_integral!(du, u, mesh::TreeMesh{2}, nonconservative_terms::True, equations,
                               volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM)
    symmetric_flux, nonconservative_flux = dg.volume_integral.volume_flux

    derivative_split = dg.basis.derivative_split
    set_diagonal_to_zero!(derivative_split) # temporarily set here, maybe move outside `rhs!`

    derivative_split = CuArray{Float64}(derivative_split)
    symmetric_flux_arr1 = CuArray{Float64}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                           size(u, 4))
    symmetric_flux_arr2 = CuArray{Float64}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                           size(u, 4))
    noncons_flux_arr1 = CuArray{Float64}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                         size(u, 4))
    noncons_flux_arr2 = CuArray{Float64}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                         size(u, 4))

    size_arr = CuArray{Float64}(undef, size(u, 2)^3, size(u, 4))

    symmetric_noncons_flux_kernel = @cuda launch=false symmetric_noncons_flux_kernel!(symmetric_flux_arr1,
                                                                                      symmetric_flux_arr2,
                                                                                      noncons_flux_arr1,
                                                                                      noncons_flux_arr2,
                                                                                      u,
                                                                                      derivative_split,
                                                                                      equations,
                                                                                      symmetric_flux,
                                                                                      nonconservative_flux)
    symmetric_noncons_flux_kernel(symmetric_flux_arr1, symmetric_flux_arr2, noncons_flux_arr1,
                                  noncons_flux_arr2, u, derivative_split, equations, symmetric_flux,
                                  nonconservative_flux;
                                  configurator_2d(symmetric_noncons_flux_kernel, size_arr)...)

    derivative_split = CuArray{Float64}(dg.basis.derivative_split) # use original `derivative_split`
    size_arr = CuArray{Float64}(undef, size(du, 1), size(du, 2)^2, size(du, 4))

    volume_integral_kernel = @cuda launch=false volume_integral_kernel!(du, derivative_split,
                                                                        symmetric_flux_arr1,
                                                                        symmetric_flux_arr2,
                                                                        noncons_flux_arr1,
                                                                        noncons_flux_arr2)
    volume_integral_kernel(du, derivative_split, symmetric_flux_arr1, symmetric_flux_arr2,
                           noncons_flux_arr1, noncons_flux_arr2;
                           configurator_3d(volume_integral_kernel, size_arr)...)

    return nothing
end

# Pack kernels for prolonging solution to interfaces
function cuda_prolong2interfaces!(u, mesh::TreeMesh{2}, equations, cache)
    neighbor_ids = CuArray{Int64}(cache.interfaces.neighbor_ids)
    orientations = CuArray{Int64}(cache.interfaces.orientations)
    interfaces_u = CuArray{Float64}(cache.interfaces.u)

    size_arr = CuArray{Float64}(undef, size(interfaces_u, 2) * size(interfaces_u, 3),
                                size(interfaces_u, 4))

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
function cuda_interface_flux!(mesh::TreeMesh{2}, nonconservative_terms::False, equations, dg::DGSEM,
                              cache)
    surface_flux = dg.surface_integral.surface_flux

    neighbor_ids = CuArray{Int64}(cache.interfaces.neighbor_ids)
    orientations = CuArray{Int64}(cache.interfaces.orientations)
    interfaces_u = CuArray{Float64}(cache.interfaces.u)
    surface_flux_arr = CuArray{Float64}(undef, 1, size(interfaces_u)[2:end]...)
    surface_flux_values = CuArray{Float64}(cache.elements.surface_flux_values)

    size_arr = CuArray{Float64}(undef, size(interfaces_u, 3), size(interfaces_u, 4))

    surface_flux_kernel = @cuda launch=false surface_flux_kernel!(surface_flux_arr, interfaces_u,
                                                                  orientations, equations,
                                                                  surface_flux)
    surface_flux_kernel(surface_flux_arr, interfaces_u, orientations, equations, surface_flux;
                        configurator_2d(surface_flux_kernel, size_arr)...)

    size_arr = CuArray{Float64}(undef, size(surface_flux_values, 1), size(interfaces_u, 3),
                                size(interfaces_u, 4))

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

function cuda_interface_flux!(mesh::TreeMesh{2}, nonconservative_terms::True, equations, dg::DGSEM,
                              cache)
    surface_flux, nonconservative_flux = dg.surface_integral.surface_flux

    neighbor_ids = CuArray{Int64}(cache.interfaces.neighbor_ids)
    orientations = CuArray{Int64}(cache.interfaces.orientations)
    interfaces_u = CuArray{Float64}(cache.interfaces.u)
    surface_flux_arr = CuArray{Float64}(undef, 1, size(interfaces_u)[2:end]...)
    noncons_left_arr = CuArray{Float64}(undef, 1, size(interfaces_u)[2:end]...)
    noncons_right_arr = CuArray{Float64}(undef, 1, size(interfaces_u)[2:end]...)
    surface_flux_values = CuArray{Float64}(cache.elements.surface_flux_values)

    size_arr = CuArray{Float64}(undef, size(interfaces_u, 3), size(interfaces_u, 4))

    surface_noncons_flux_kernel = @cuda launch=false surface_noncons_flux_kernel!(surface_flux_arr,
                                                                                  interfaces_u,
                                                                                  noncons_left_arr,
                                                                                  noncons_right_arr,
                                                                                  orientations,
                                                                                  equations,
                                                                                  surface_flux,
                                                                                  nonconservative_flux)
    surface_noncons_flux_kernel(surface_flux_arr, interfaces_u, noncons_left_arr, noncons_right_arr,
                                orientations, equations, surface_flux, nonconservative_flux;
                                configurator_2d(surface_noncons_flux_kernel, size_arr)...)

    size_arr = CuArray{Float64}(undef, size(surface_flux_values, 1), size(interfaces_u, 3),
                                size(interfaces_u, 4))

    interface_flux_kernel = @cuda launch=false interface_flux_kernel!(surface_flux_values,
                                                                      surface_flux_arr,
                                                                      noncons_left_arr,
                                                                      noncons_right_arr,
                                                                      neighbor_ids, orientations)
    interface_flux_kernel(surface_flux_values, surface_flux_arr, noncons_left_arr,
                          noncons_right_arr, neighbor_ids, orientations;
                          configurator_3d(interface_flux_kernel, size_arr)...)

    cache.elements.surface_flux_values = surface_flux_values # copy back to host automatically

    return nothing
end

# Dummy function for asserting boundaries
function cuda_prolong2boundaries!(u, mesh::TreeMesh{2},
                                  boundary_condition::BoundaryConditionPeriodic, equations, cache)
    @assert iszero(length(cache.boundaries.orientations))
end

# Pack kernels for prolonging solution to boundaries
function cuda_prolong2boundaries!(u, mesh::TreeMesh{2}, boundary_conditions::NamedTuple, equations,
                                  cache)
    neighbor_ids = CuArray{Int64}(cache.boundaries.neighbor_ids)
    neighbor_sides = CuArray{Int64}(cache.boundaries.neighbor_sides)
    orientations = CuArray{Int64}(cache.boundaries.orientations)
    boundaries_u = CuArray{Float64}(cache.boundaries.u)

    size_arr = CuArray{Float64}(undef, size(boundaries_u, 2) * size(boundaries_u, 3),
                                size(boundaries_u, 4))

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
function cuda_boundary_flux!(t, mesh::TreeMesh{2}, boundary_condition::BoundaryConditionPeriodic,
                             equations, dg::DGSEM, cache)
    @assert iszero(length(cache.boundaries.orientations))
end

# Pack kernels for calculating boundary fluxes
function cuda_boundary_flux!(t, mesh::TreeMesh{2}, boundary_conditions::NamedTuple, equations,
                             dg::DGSEM, cache)
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
    boundary_arr = CuArray{Int64}(firsts[1]:lasts[4])
    indices_arr = CuArray{Int64}([firsts[1], firsts[2], firsts[3], firsts[4]])
    boundary_conditions_callable = replace_boundary_conditions(boundary_conditions)

    size_arr = CuArray{Float64}(undef, size(surface_flux_values, 2), length(boundary_arr))

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
                         configurator_2d(boundary_flux_kernel, size_arr)...)

    cache.elements.surface_flux_values = surface_flux_values # copy back to host automatically

    return nothing
end

# Dummy function for asserting mortars
function cuda_prolong2mortars!(u, mesh::TreeMesh{2}, cache_mortars::False, dg::DGSEM, cache)
    @assert iszero(length(cache.mortars.orientations))
end

# Pack kernels for prolonging solution to mortars
function cuda_prolong2mortars!(u, mesh::TreeMesh{2}, cache_mortars::True, dg::DGSEM, cache)
    neighbor_ids = CuArray{Int64}(cache.mortars.neighbor_ids)
    large_sides = CuArray{Int64}(cache.mortars.large_sides)
    orientations = CuArray{Int64}(cache.mortars.orientations)

    u_upper = zero(CuArray{Float64}(cache.mortars.u_upper)) # NaN to zero
    u_lower = zero(CuArray{Float64}(cache.mortars.u_lower)) # NaN to zero

    forward_upper = CuArray{Float64}(dg.mortar.forward_upper)
    forward_lower = CuArray{Float64}(dg.mortar.forward_lower)

    size_arr = CuArray{Float64}(undef, size(u_upper, 2), size(u_upper, 3), size(u_upper, 4))

    prolong_mortars_small2small_kernel = @cuda launch=false prolong_mortars_small2small_kernel!(u_upper,
                                                                                                u_lower,
                                                                                                u,
                                                                                                neighbor_ids,
                                                                                                large_sides,
                                                                                                orientations)
    prolong_mortars_small2small_kernel(u_upper, u_lower, u, neighbor_ids, large_sides, orientations;
                                       configurator_3d(prolong_mortars_small2small_kernel,
                                                       size_arr)...)

    prolong_mortars_large2small_kernel = @cuda launch=false prolong_mortars_large2small_kernel!(u_upper,
                                                                                                u_lower,
                                                                                                u,
                                                                                                forward_upper,
                                                                                                forward_lower,
                                                                                                neighbor_ids,
                                                                                                large_sides,
                                                                                                orientations)
    prolong_mortars_large2small_kernel(u_upper, u_lower, u, forward_upper, forward_lower,
                                       neighbor_ids, large_sides, orientations;
                                       configurator_3d(prolong_mortars_large2small_kernel,
                                                       size_arr)...)

    cache.mortars.u_upper = u_upper # copy back to host automatically
    cache.mortars.u_lower = u_lower # copy back to host automatically

    return nothing
end

# Dummy function for asserting mortar fluxes
function cuda_mortar_flux!(mesh::TreeMesh{2}, cache_mortars::False, nonconservative_terms,
                           equations, dg::DGSEM, cache)
    @assert iszero(length(cache.mortars.orientations))
end

# Pack kernels for calculating mortar fluxes
function cuda_mortar_flux!(mesh::TreeMesh{2}, cache_mortars::True, nonconservative_terms::False,
                           equations, dg::DGSEM, cache)
    surface_flux = dg.surface_integral.surface_flux

    neighbor_ids = CuArray{Int64}(cache.mortars.neighbor_ids)
    large_sides = CuArray{Int64}(cache.mortars.large_sides)
    orientations = CuArray{Int64}(cache.mortars.orientations)

    u_upper = CuArray{Float64}(cache.mortars.u_upper)
    u_lower = CuArray{Float64}(cache.mortars.u_lower)
    reverse_upper = CuArray{Float64}(dg.mortar.reverse_upper)
    reverse_lower = CuArray{Float64}(dg.mortar.reverse_lower)

    surface_flux_values = CuArray{Float64}(cache.elements.surface_flux_values)
    tmp_surface_flux_values = zero(similar(surface_flux_values))

    fstar_upper = CuArray{Float64}(undef, size(u_upper, 2), size(u_upper, 3), length(orientations))
    fstar_lower = CuArray{Float64}(undef, size(u_upper, 2), size(u_upper, 3), length(orientations))

    size_arr = CuArray{Float64}(undef, size(u_upper, 3), length(orientations))

    mortar_flux_kernel = @cuda launch=false mortar_flux_kernel!(fstar_upper, fstar_lower, u_upper,
                                                                u_lower, orientations, equations,
                                                                surface_flux)
    mortar_flux_kernel(fstar_upper, fstar_lower, u_upper, u_lower, orientations, equations,
                       surface_flux;
                       configurator_2d(mortar_flux_kernel, size_arr)...)

    size_arr = CuArray{Float64}(undef, size(surface_flux_values, 1), size(surface_flux_values, 2),
                                length(orientations))

    mortar_flux_copy_to_kernel = @cuda launch=false mortar_flux_copy_to_kernel!(surface_flux_values,
                                                                                tmp_surface_flux_values,
                                                                                fstar_upper,
                                                                                fstar_lower,
                                                                                reverse_upper,
                                                                                reverse_lower,
                                                                                neighbor_ids,
                                                                                large_sides,
                                                                                orientations)
    mortar_flux_copy_to_kernel(surface_flux_values, tmp_surface_flux_values, fstar_upper,
                               fstar_lower,
                               reverse_upper, reverse_lower, neighbor_ids, large_sides,
                               orientations;
                               configurator_3d(mortar_flux_copy_to_kernel, size_arr)...)

    cache.elements.surface_flux_values = surface_flux_values # copy back to host automatically
end

function cuda_mortar_flux!(mesh::TreeMesh{2}, cache_mortars::True, nonconservative_terms::True,
                           equations, dg::DGSEM, cache)
end

# Pack kernels for calculating surface integrals
function cuda_surface_integral!(du, mesh::TreeMesh{2}, equations, dg::DGSEM, cache)
    factor_arr = CuArray{Float64}([
                                      dg.basis.boundary_interpolation[1, 1],
                                      dg.basis.boundary_interpolation[size(du, 2), 2]
                                  ])
    surface_flux_values = CuArray{Float64}(cache.elements.surface_flux_values)

    size_arr = CuArray{Float64}(undef, size(du, 1), size(du, 2)^2, size(du, 4))

    surface_integral_kernel = @cuda launch=false surface_integral_kernel!(du, factor_arr,
                                                                          surface_flux_values,
                                                                          equations)
    surface_integral_kernel(du, factor_arr, surface_flux_values, equations;
                            configurator_3d(surface_integral_kernel, size_arr)...)

    return nothing
end

# Pack kernels for applying Jacobian to reference element
function cuda_jacobian!(du, mesh::TreeMesh{2}, equations, cache)
    inverse_jacobian = CuArray{Float64}(cache.elements.inverse_jacobian)

    size_arr = CuArray{Float64}(undef, size(du, 1), size(du, 2)^2, size(du, 4))

    jacobian_kernel = @cuda launch=false jacobian_kernel!(du, inverse_jacobian, equations)
    jacobian_kernel(du, inverse_jacobian, equations; configurator_3d(jacobian_kernel, size_arr)...)

    return nothing
end

# Dummy function returning nothing              
function cuda_sources!(du, u, t, source_terms::Nothing, equations::AbstractEquations{2}, cache)
    return nothing
end

# Pack kernels for calculating source terms 
function cuda_sources!(du, u, t, source_terms, equations::AbstractEquations{2}, cache)
    node_coordinates = CuArray{Float64}(cache.elements.node_coordinates)

    size_arr = CuArray{Float64}(undef, size(u, 2)^2, size(u, 4))

    source_terms_kernel = @cuda launch=false source_terms_kernel!(du, u, node_coordinates, t,
                                                                  equations, source_terms)
    source_terms_kernel(du, u, node_coordinates, t, equations, source_terms;
                        configurator_2d(source_terms_kernel, size_arr)...)

    return nothing
end

# Put everything together into a single function 
# Ref: `rhs!` function in Trixi.jl

function rhs_gpu!(du_cpu, u_cpu, t, mesh::TreeMesh{2}, equations, boundary_conditions,
                  source_terms::Source, dg::DGSEM, cache) where {Source}
    du, u = copy_to_device!(du_cpu, u_cpu)

    cuda_volume_integral!(du, u, mesh, have_nonconservative_terms(equations), equations,
                          dg.volume_integral, dg)

    cuda_prolong2interfaces!(u, mesh, equations, cache)

    cuda_interface_flux!(mesh, have_nonconservative_terms(equations), equations, dg, cache)

    cuda_prolong2boundaries!(u, mesh, boundary_conditions, equations, cache)

    cuda_boundary_flux!(t, mesh, boundary_conditions, equations, dg, cache)

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
