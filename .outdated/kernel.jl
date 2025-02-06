# Outdated kernels that are no longer used in the current implementation due to optimizations.

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
#     forward_upper = dg.mortar.forward_upper
#     forward_lower = dg.mortar.forward_lower

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
#     reverse_upper = dg.mortar.reverse_upper
#     reverse_lower = dg.mortar.reverse_lower

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
#     reverse_upper = dg.mortar.reverse_upper
#     reverse_lower = dg.mortar.reverse_lower

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
