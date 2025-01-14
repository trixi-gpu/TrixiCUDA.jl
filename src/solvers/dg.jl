# The `wrap_array` function in Trixi.jl is not compatible with GPU arrays,
# so here we adapt `wrap_array` to work with GPU arrays.
@inline function wrap_array(u_ode::CuArray, mesh::AbstractMesh, equations,
                            dg::DGSEM, cache)
    # TODO: Assert array length before calling `reshape`
    u_ode = reshape(u_ode, nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))...,
                    nelements(dg, cache))

    return u_ode
end

# The `wrap_array_native` function in Trixi.jl is not compatible with GPU arrays,
# so here we adapt `wrap_array_native` to work with GPU arrays.
@inline function wrap_array_native(u_ode::CuArray, mesh::AbstractMesh, equations,
                                   dg::DGSEM, cache)
    # TODO: Assert array length before calling `reshape`
    u_ode = reshape(u_ode, nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))...,
                    nelements(dg, cache))

    return u_ode
end

# Do we really need to compute the coefficients on the GPU, and do we need to
# initialize `du` and `u` with a 1D shape, as Trixi.jl does?
# Note that the above questions also relate to whether using `wrap_array` and `wrap_array_native`.
############################################################################## 
# Kernel for computing the coefficients for 1D problems
function compute_coefficients_kernel!(u, node_coordinates, func::Any, t,
                                      equations::AbstractEquations{1})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2) && k <= size(u, 3))
        x_node = get_node_coords(node_coordinates, equations, j, k)

        if j == 1 # bad
            @inbounds x_node = SVector(nextfloat(x_node[1]))
        elseif j == size(u, 2) # bad
            @inbounds x_node = SVector(prevfloat(x_node[1]))
        end

        u_node = func(x_node, t, equations)

        for ii in axes(u, 1)
            @inbounds u[ii, j, k] = u_node[ii]
        end
    end

    return nothing
end

# Kernel for computing the coefficients for 2D problems
function compute_coefficients_kernel!(u, node_coordinates, func::Any, t,
                                      equations::AbstractEquations{2})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^2 && k <= size(u, 4))
        j1 = div(j - 1, size(u, 2)) + 1
        j2 = rem(j - 1, size(u, 2)) + 1

        x_node = get_node_coords(node_coordinates, equations, j1, j2, k)

        u_node = func(x_node, t, equations)

        for ii in axes(u, 1)
            @inbounds u[ii, j1, j2, k] = u_node[ii]
        end
    end

    return nothing
end

# Kernel for computing the coefficients for 3D problems
function compute_coefficients_kernel!(u, node_coordinates, func::Any, t,
                                      equations::AbstractEquations{3})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^3 && k <= size(u, 5))
        u2 = size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        x_node = get_node_coords(node_coordinates, equations, j1, j2, j3, k)

        u_node = func(x_node, t, equations)

        for ii in axes(u, 1)
            @inbounds u[ii, j1, j2, j3, k] = u_node[ii]
        end
    end

    return nothing
end

# Call kernels to compute the coefficients for 1D problems
function compute_coefficients_gpu(u, func, t, mesh::AbstractMesh{1}, equations, dg::DGSEM, cache)
    node_coordinates = cache.elements.node_coordinates

    compute_coefficients_kernel = @cuda launch=false compute_coefficients_kernel!(u,
                                                                                  node_coordinates,
                                                                                  func, t,
                                                                                  equations)
    compute_coefficients_kernel(u, node_coordinates, func, t, equations;
                                kernel_configurator_2d(compute_coefficients_kernel, size(u, 2),
                                                       size(u, 3))...)

    return u
end

# Call kernels to compute the coefficients for 2D problems
function compute_coefficients_gpu(u, func, t, mesh::AbstractMesh{2}, equations, dg::DGSEM, cache)
    node_coordinates = cache.elements.node_coordinates

    compute_coefficients_kernel = @cuda launch=false compute_coefficients_kernel!(u,
                                                                                  node_coordinates,
                                                                                  func, t,
                                                                                  equations)
    compute_coefficients_kernel(u, node_coordinates, func, t, equations;
                                kernel_configurator_2d(compute_coefficients_kernel, size(u, 2)^2,
                                                       size(u, 4))...)

    return u
end

# Call kernels to compute the coefficients for 2D problems
function compute_coefficients_gpu(u, func, t, mesh::AbstractMesh{3}, equations, dg::DGSEM, cache)
    node_coordinates = cache.elements.node_coordinates

    compute_coefficients_kernel = @cuda launch=false compute_coefficients_kernel!(u,
                                                                                  node_coordinates,
                                                                                  func, t,
                                                                                  equations)
    compute_coefficients_kernel(u, node_coordinates, func, t, equations;
                                kernel_configurator_2d(compute_coefficients_kernel, size(u, 2)^3,
                                                       size(u, 5))...)

    return u
end
