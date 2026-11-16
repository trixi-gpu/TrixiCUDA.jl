# Get the polynomial degree from the solver basis.
@inline polydeg(dg::DGSEM{<:LobattoLegendreBasisGPU}) = polydeg(dg.basis)

# TrixiCUDA.jl keeps the solution in the `(nvariables, nnodes, ..., nnodes, nelements)` layout
# used by the kernels (see `compute_coefficients_gpu`), so the RHS does not wrap its arrays.
# This method is still needed because Trixi.jl's callbacks (e.g., `StepsizeCallback` and
# `AnalysisCallback`) call `wrap_array` on the integrator state. For arrays that already
# have this layout, `reshape` returns an array of the same shape without copying.
@inline function wrap_array(u_ode::AbstractGPUArray, mesh::AbstractMesh, equations,
                            dg::DGSEM, cache)
    reshape(u_ode, nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))...,
            nelements(dg, cache))
end

# Callbacks copy the GPU state to a host Array that already has the expected layout.
# Return it directly to support Trixi's existing callback interface.
# TODO: Keep more callback computations on the GPU and improve the interface to
# reduce host transfers and avoid pass-through adapters like this one.
@inline function wrap_array_native(u_ode::Array, mesh::TreeMesh, equations,
                                   dg::DGSEM{<:LobattoLegendreBasisGPU}, cache)
    return u_ode
end

# FIXME: This is a temporary workaround to avoid scalar indexing issue.
function volume_jacobian_tmp(element, mesh::TreeMesh, cache)
    inverse_jacobian = Array(cache.elements.inverse_jacobian)
    return inv(inverse_jacobian[element])^ndims(mesh)
end

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

    # Keep captured types in generated initial conditions concrete on Julia 1.10/1.11.
    compute_coefficients_kernel = @cuda launch=false always_inline=true compute_coefficients_kernel!(u,
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

    # Keep captured types in generated initial conditions concrete on Julia 1.10/1.11.
    compute_coefficients_kernel = @cuda launch=false always_inline=true compute_coefficients_kernel!(u,
                                                                                                     node_coordinates,
                                                                                                     func, t,
                                                                                                     equations)
    compute_coefficients_kernel(u, node_coordinates, func, t, equations;
                                kernel_configurator_2d(compute_coefficients_kernel, size(u, 2)^2,
                                                       size(u, 4))...)

    return u
end

# Call kernels to compute the coefficients for 3D problems
function compute_coefficients_gpu(u, func, t, mesh::AbstractMesh{3}, equations, dg::DGSEM, cache)
    node_coordinates = cache.elements.node_coordinates

    compute_coefficients_kernel = @cuda launch=false always_inline=true compute_coefficients_kernel!(u,
                                                                                                     node_coordinates,
                                                                                                     func, t,
                                                                                                     equations)
    compute_coefficients_kernel(u, node_coordinates, func, t, equations;
                                kernel_configurator_2d(compute_coefficients_kernel, size(u, 2)^3,
                                                       size(u, 5))...)

    return u
end

# Helper functions to call the check to conservative terms for kernels.
function calc_volume_integral!(du, u, mesh::TreeMesh, equations,
                               volume_integral, dg::DGSEM, cache)
    return calc_volume_integral!(du, u, mesh, have_nonconservative_terms(equations), equations,
                                 volume_integral, dg, cache)
end
function calc_interface_flux!(surface_flux_values, mesh::TreeMesh, equations,
                              surface_integral, dg::DGSEM, cache)
    return calc_interface_flux!(surface_flux_values, mesh,
                                have_nonconservative_terms(equations), equations,
                                surface_integral, dg, cache)
end
function calc_boundary_flux!(cache, t,
                             boundary_conditions::NamedTuple,
                             mesh::Union{TreeMesh{1}, TreeMesh{2}}, equations,
                             surface_integral, dg::DGSEM)
    return calc_boundary_flux!(cache, t, boundary_conditions, mesh,
                               have_nonconservative_terms(equations), equations,
                               surface_integral, dg)
end
function calc_mortar_flux!(surface_flux_values, mesh::Union{TreeMesh{2}, TreeMesh{3}}, equations,
                           mortar_l2, surface_integral, dg::DGSEM, cache)
    return calc_mortar_flux!(surface_flux_values, mesh,
                             have_nonconservative_terms(equations), equations,
                             mortar_l2, surface_integral, dg, cache)
end
