# Everything related to a DG semidiscretization in 1D

# Functions that end with `_kernel` are CUDA kernels that are going to be launched by 
# the @cuda macro with parameters from the kernel configurator. They are purely run on 
# the device (i.e., GPU).

# Kernel for calculating fluxes along normal direction
function flux_kernel!(flux_arr, u, equations::AbstractEquations{1}, flux::Function)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2) && k <= size(u, 3))
        u_node = get_node_vars(u, equations, j, k)

        flux_node = flux(u_node, 1, equations)

        @inbounds begin
            for ii in axes(u, 1)
                flux_arr[ii, j, k] = flux_node[ii]
            end
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
        @inbounds begin
            for ii in axes(du, 2)
                du[i, j, k] += derivative_dhat[j, ii] * flux_arr[i, ii, k]
            end
        end
    end

    return nothing
end

# Kernel for calculating volume fluxes
function volume_flux_kernel!(volume_flux_arr, u, equations::AbstractEquations{1},
                             volume_flux::Function)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^2 && k <= size(u, 3))
        j1 = div(j - 1, size(u, 2)) + 1
        j2 = rem(j - 1, size(u, 2)) + 1

        u_node = get_node_vars(u, equations, j1, k)
        u_node1 = get_node_vars(u, equations, j2, k)

        volume_flux_node = volume_flux(u_node, u_node1, 1, equations)

        @inbounds begin
            for ii in axes(u, 1)
                volume_flux_arr[ii, j1, j2, k] = volume_flux_node[ii]
            end
        end
    end

    return nothing
end

# Kernel for calculating volume integrals
function volume_integral_kernel!(du, derivative_split, volume_flux_arr)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        @inbounds begin
            for ii in axes(du, 2)
                du[i, j, k] += derivative_split[j, ii] * volume_flux_arr[i, j, ii, k]
            end
        end
    end

    return nothing
end

# Kernel for prolonging two interfaces
function prolong_interfaces_kernel!(interfaces_u, u, neighbor_ids, equations::AbstractEquations{1})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(interfaces_u, 2) && k <= size(interfaces_u, 3))
        left_element = neighbor_ids[1, k]
        right_element = neighbor_ids[2, k]

        @inbounds begin
            interfaces_u[1, j, k] = u[j, size(u, 2), left_element]
            interfaces_u[2, j, k] = u[j, 1, right_element]
        end
    end

    return nothing
end

# Kernel for calculating surface fluxes 
function surface_flux_kernel!(surface_flux_arr, interfaces_u, equations::AbstractEquations{1},
                              surface_flux::Any)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if (k <= size(surface_flux_arr, 3))
        u_ll, u_rr = get_surface_node_vars(interfaces_u, equations, k)

        surface_flux_node = surface_flux(u_ll, u_rr, 1, equations)

        @inbounds begin
            for jj in axes(surface_flux_arr, 2)
                surface_flux_arr[1, jj, k] = surface_flux_node[jj]
            end
        end
    end

    return nothing
end

# Kernel for setting interface fluxes
function interface_flux_kernel!(surface_flux_values, surface_flux_arr, neighbor_ids,
                                equations::AbstractEquations{1})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (i <= size(surface_flux_values, 1) && k <= size(surface_flux_arr, 3))
        left_id = neighbor_ids[1, k]
        right_id = neighbor_ids[2, k]

        @inbounds begin
            surface_flux_values[i, 2, left_id] = surface_flux_arr[1, i, k]
            surface_flux_values[i, 1, right_id] = surface_flux_arr[1, i, k]
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
                              source_terms::Function)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(du, 2) && k <= size(du, 3))
        u_local = get_node_vars(u, equations, j, k)
        x_local = get_node_coords(node_coordinates, equations, j, k)

        source_terms_node = source_terms(u_local, x_local, t, equations)

        @inbounds begin
            for ii in axes(du, 1)
                du[ii, j, k] += source_terms_node[ii]
            end
        end
    end

    return nothing
end

# Functions that begin with `cuda_` are the functions that pack CUDA kernels together to do 
# partial work in semidiscretization. They are used to invoke kernels from the host (i.e., CPU) 
# and run them on the device (i.e., GPU).

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{1}, nonconservative_terms,
                               equations, volume_integral::VolumeIntegralWeakForm, dg::DGSEM)
    derivative_dhat = CuArray{Float64}(dg.basis.derivative_dhat)
    flux_arr = similar(u)

    size_arr = CuArray{Float64}(undef, size(u, 2), size(u, 3))

    flux_kernel = @cuda launch=false flux_kernel!(flux_arr, u, equations, flux)
    flux_kernel(flux_arr, u, equations, flux; configurator_2d(flux_kernel, size_arr)...)

    weak_form_kernel = @cuda launch=false weak_form_kernel!(du, derivative_dhat, flux_arr)
    weak_form_kernel(du, derivative_dhat, flux_arr; configurator_3d(weak_form_kernel, du)...)

    return nothing
end

function cuda_volume_integral!(du, u, mesh::TreeMesh{1}, nonconservative_terms::False,
                               equations, volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGSEM)
    volume_flux = volume_integral.volume_flux

    derivative_split = CuArray{Float64}(dg.basis.derivative_split)
    volume_flux_arr = CuArray{Float64}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 3))

    size_arr = CuArray{Float64}(undef, size(u, 2)^2, size(u, 3))

    volume_flux_kernel = @cuda launch=false volume_flux_kernel!(volume_flux_arr, u, equations,
                                                                volume_flux)
    volume_flux_kernel(volume_flux_arr, u, equations, volume_flux;
                       configurator_2d(volume_flux_kernel, size_arr)...,)

    volume_integral_kernel = @cuda launch=false volume_integral_kernel!(du, derivative_split,
                                                                        volume_flux_arr)
    volume_integral_kernel(du, derivative_split, volume_flux_arr;
                           configurator_3d(volume_integral_kernel, du)...,)

    return nothing
end

# Pack kernels for prolonging solution to interfaces
function cuda_prolong2interfaces!(u, mesh::TreeMesh{1}, equations, cache)
    neighbor_ids = CuArray{Int64}(cache.interfaces.neighbor_ids)
    interfaces_u = CuArray{Float64}(cache.interfaces.u)

    size_arr = CuArray{Float64}(undef, size(interfaces_u, 2), size(interfaces_u, 3))

    prolong_interfaces_kernel = @cuda launch=false prolong_interfaces_kernel!(interfaces_u, u,
                                                                              neighbor_ids,
                                                                              equations)
    prolong_interfaces_kernel(interfaces_u, u, neighbor_ids, equations;
                              configurator_2d(prolong_interfaces_kernel, size_arr)...,)

    cache.interfaces.u = interfaces_u  # copy back to host automatically

    return nothing
end

# Pack kernels for calculating interface fluxes
function cuda_interface_flux!(mesh::TreeMesh{1}, nonconservative_terms::False, equations, dg::DGSEM,
                              cache)
    surface_flux = dg.surface_integral.surface_flux

    neighbor_ids = CuArray{Int64}(cache.interfaces.neighbor_ids)
    interfaces_u = CuArray{Float64}(cache.interfaces.u)
    surface_flux_arr = CuArray{Float64}(undef, 1, size(interfaces_u)[2:end]...)
    surface_flux_values = CuArray{Float64}(cache.elements.surface_flux_values)

    size_arr = CuArray{Float64}(undef, size(interfaces_u, 3))

    surface_flux_kernel = @cuda launch=false surface_flux_kernel!(surface_flux_arr, interfaces_u,
                                                                  equations, surface_flux)
    surface_flux_kernel(surface_flux_arr, interfaces_u, equations, surface_flux;
                        configurator_1d(surface_flux_kernel, size_arr)...,)

    size_arr = CuArray{Float64}(undef, size(surface_flux_values, 1), size(interfaces_u, 3))

    interface_flux_kernel = @cuda launch=false interface_flux_kernel!(surface_flux_values,
                                                                      surface_flux_arr,
                                                                      neighbor_ids, equations)
    interface_flux_kernel(surface_flux_values, surface_flux_arr, neighbor_ids, equations;
                          configurator_2d(interface_flux_kernel, size_arr)...,)

    cache.elements.surface_flux_values = surface_flux_values # copy back to host automatically

    return nothing
end

# Dummy function for asserting boundaries
function cuda_prolong2boundaries!(u, mesh::TreeMesh{1},
                                  boundary_condition::BoundaryConditionPeriodic, cache)
    @assert iszero(length(cache.boundaries.orientations))
end

# Dummy function for asserting boundary fluxes
function cuda_boundary_flux!(t, mesh::TreeMesh{1}, boundary_condition::BoundaryConditionPeriodic,
                             equations, dg::DGSEM, cache)
    @assert iszero(length(cache.boundaries.orientations))
end

# Pack kernels for calculating surface integrals
function cuda_surface_integral!(du, mesh::TreeMesh{1}, equations, dg::DGSEM, cache)
    # FIXME: Check `surface_integral`
    factor_arr = CuArray{Float64}([
                                      dg.basis.boundary_interpolation[1, 1],
                                      dg.basis.boundary_interpolation[size(du, 2), 2]
                                  ])
    surface_flux_values = CuArray{Float64}(cache.elements.surface_flux_values)

    size_arr = CuArray{Float64}(undef, size(du, 1), size(du, 2), size(du, 3))

    surface_integral_kernel = @cuda launch=false surface_integral_kernel!(du, factor_arr,
                                                                          surface_flux_values,
                                                                          equations)
    surface_integral_kernel(du, factor_arr, surface_flux_values, equations;
                            configurator_3d(surface_integral_kernel, size_arr)...,)

    return nothing
end

# Pack kernels for applying Jacobian to reference element
function cuda_jacobian!(du, mesh::TreeMesh{1}, equations, cache)
    inverse_jacobian = CuArray{Float64}(cache.elements.inverse_jacobian)

    jacobian_kernel = @cuda launch=false jacobian_kernel!(du, inverse_jacobian, equations)
    jacobian_kernel(du, inverse_jacobian, equations; configurator_3d(jacobian_kernel, du)...)

    return nothing
end

# Dummy function returning nothing             
function cuda_sources!(du, u, t, source_terms::Nothing, equations::AbstractEquations{1}, cache)
    return nothing
end

# Pack kernels for calculating source terms 
function cuda_sources!(du, u, t, source_terms, equations::AbstractEquations{1}, cache)
    node_coordinates = CuArray{Float32}(cache.elements.node_coordinates)

    size_arr = CuArray{Float32}(undef, size(du, 2), size(du, 3))

    source_terms_kernel = @cuda launch=false source_terms_kernel!(du, u, node_coordinates, t,
                                                                  equations, source_terms)
    source_terms_kernel(du, u, node_coordinates, t, equations, source_terms;
                        configurator_2d(source_terms_kernel, size_arr)...,)

    return nothing
end

# Put everything together into a single function 
# Ref: `rhs!` function in Trixi.jl

function rhs_gpu!(du_cpu, u_cpu, t, mesh::TreeMesh{1}, equations, initial_condition,
                  boundary_conditions, source_terms::Source, dg::DGSEM, cache) where {Source}
    du, u = copy_to_device!(du_cpu, u_cpu)

    cuda_volume_integral!(du, u, mesh, have_nonconservative_terms(equations), equations,
                          dg.volume_integral, dg)

    cuda_prolong2interfaces!(u, mesh, equations, cache)

    cuda_interface_flux!(mesh, have_nonconservative_terms(equations), equations, dg, cache)

    cuda_prolong2boundaries!(u, mesh, boundary_conditions, cache)

    cuda_boundary_flux!(t, mesh, boundary_conditions, equations, dg, cache)

    cuda_surface_integral!(du, mesh, equations, dg, cache)

    cuda_jacobian!(du, mesh, equations, cache)

    cuda_sources!(du, u, t, source_terms, equations, cache)

    du_computed, _ = copy_to_host!(du, u)
    du_cpu .= du_computed

    return nothing
end
