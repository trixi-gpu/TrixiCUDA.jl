# Everything related to a DG semidiscretization in 1D

# TODO: Please check whether `equations::AbstractEquations{1}` is needed for each function here!!
# Functions end with `_kernel` are CUDA kernels that are going to be launed by the `@cuda` macro.

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

# Kernel for prolonging two interfaces
function prolong_interfaces_kernel!(interfaces_u, u, neighbor_ids)
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
function surface_flux_kernel!(surface_flux_arr, interfaces_u,
                              equations::AbstractEquations{1}, surface_flux::Any)
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
function interface_flux_kernel!(surface_flux_values, surface_flux_arr, neighbor_ids)
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

# Functions begin with `cuda_` are the functions that pack CUDA kernels together, 
# calling Tthem from the host (i.e., CPU) and running them on the device (i.e., GPU).

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{1}, nonconservative_terms, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::DGSEM)
    derivative_dhat = CuArray{Float32}(dg.basis.derivative_dhat)
    flux_arr = similar(u)

    size_arr = CuArray{Float32}(undef, size(u, 2), size(u, 3))

    flux_kernel = @cuda launch=false flux_kernel!(flux_arr, u, equations, flux)
    flux_kernel(flux_arr, u, equations, flux; configurator_2d(flux_kernel, size_arr)...)

    weak_form_kernel = @cuda launch=false weak_form_kernel!(du, derivative_dhat, flux_arr)
    weak_form_kernel(du, derivative_dhat, flux_arr;
                     configurator_3d(weak_form_kernel, du)...,)

    return nothing
end

# Pack kernels for prolonging solution to interfaces
function cuda_prolong2interfaces!(u, mesh::TreeMesh{1}, cache)
    neighbor_ids = CuArray{Int64}(cache.interfaces.neighbor_ids)
    interfaces_u = CuArray{Float32}(cache.interfaces.u)

    size_arr = CuArray{Float32}(undef, size(interfaces_u, 2), size(interfaces_u, 3))

    prolong_interfaces_kernel = @cuda launch=false prolong_interfaces_kernel!(interfaces_u,
                                                                              u,
                                                                              neighbor_ids)
    prolong_interfaces_kernel(interfaces_u, u, neighbor_ids;
                              configurator_2d(prolong_interfaces_kernel, size_arr)...,)

    cache.interfaces.u = interfaces_u  # copy back to host automatically

    return nothing
end

# Pack kernels for calculating interface fluxes
function cuda_interface_flux!(mesh::TreeMesh{1}, nonconservative_terms::False, equations,
                              dg::DGSEM, cache)
    surface_flux = dg.surface_integral.surface_flux

    neighbor_ids = CuArray{Int64}(cache.interfaces.neighbor_ids)
    interfaces_u = CuArray{Float32}(cache.interfaces.u)
    surface_flux_arr = CuArray{Float32}(undef, 1, size(interfaces_u)[2:end]...)
    surface_flux_values = CuArray{Float32}(cache.elements.surface_flux_values)

    size_arr = CuArray{Float32}(undef, size(interfaces_u, 3))

    surface_flux_kernel = @cuda launch=false surface_flux_kernel!(surface_flux_arr,
                                                                  interfaces_u, equations,
                                                                  surface_flux)
    surface_flux_kernel(surface_flux_arr, interfaces_u, equations, surface_flux;
                        configurator_1d(surface_flux_kernel, size_arr)...,)

    size_arr = CuArray{Float32}(undef, size(surface_flux_values, 1), size(interfaces_u, 3))

    interface_flux_kernel = @cuda launch=false interface_flux_kernel!(surface_flux_values,
                                                                      surface_flux_arr,
                                                                      neighbor_ids)
    interface_flux_kernel(surface_flux_values, surface_flux_arr, neighbor_ids;
                          configurator_2d(interface_flux_kernel, size_arr)...,)

    cache.elements.surface_flux_values = surface_flux_values # copy back to host automatically

    return nothing
end
