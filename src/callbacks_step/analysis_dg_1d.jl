function integrate_via_indices(func::Func, u,
                               mesh::TreeMesh{1}, equations, dg::DG, cache,
                               args...; normalize = true) where {Func}
    (; weights) = dg.basis

    # Initialize integral with zeros of the right shape
    integral = zero(func(u, 1, 1, equations, dg, args...))

    # Use quadrature to numerically integrate over entire domain (origin calls `@batch`)
    for element in eachelement(dg, cache)
        # FIXME: This is a temporary workaround to avoid the scalar indexing issue.
        volume_jacobian_ = volume_jacobian_tmp(element, mesh, cache)
        for i in eachnode(dg)
            integral += volume_jacobian_ * weights[i] *
                        func(u, i, element, equations, dg, args...)
        end
    end

    # Normalize with total volume
    if normalize
        integral = integral / total_volume(mesh)
    end

    # FIXME: This is a temporary workaround to avoid the scalar indexing issue.
    if integral isa Number
        integral = [integral]
    else
        integral = Array(integral)
    end

    return integral
end

function integrate(func::Func, u,
                   mesh::TreeMesh{1},
                   equations, dg::DG, cache; normalize = true) where {Func}
    integrate_via_indices(u, mesh, equations, dg, cache;
                          normalize = normalize) do u, i, element, equations, dg
        u_local = get_node_vars_view(u, equations, dg, i, element)  # call view to avoid scalar indexing
        return func(u_local, equations)
    end
end

function calc_error_norms(func, u, t, analyzer,
                          mesh::TreeMesh{1}, equations, initial_condition,
                          dg::DG, cache, cache_analysis)
    (; vandermonde, weights) = analyzer
    (; node_coordinates) = cache.elements
    (; u_local, x_local) = cache_analysis

    # FIXME: This is a temporary workaround to avoid the scalar indexing issue.
    node_coordinates = Array(node_coordinates)
    u = Array(u)

    # Set up data structures
    l2_error = zero(func(get_node_vars(u, equations, dg, 1, 1), equations))
    linf_error = copy(l2_error)

    # Iterate over all elements for error calculations
    for element in eachelement(dg, cache)
        # Interpolate solution and node locations to analysis nodes
        multiply_dimensionwise!(u_local, vandermonde, view(u, :, :, element))
        multiply_dimensionwise!(x_local, vandermonde,
                                view(node_coordinates, :, :, element))

        # Calculate errors at each analysis node
        # FIXME: This is a temporary workaround to avoid the scalar indexing issue.
        volume_jacobian_ = volume_jacobian_tmp(element, mesh, cache)

        for i in eachnode(analyzer)
            u_exact = initial_condition(get_node_coords(x_local, equations, dg, i), t,
                                        equations)
            diff = func(u_exact, equations) -
                   func(get_node_vars(u_local, equations, dg, i), equations)
            l2_error += diff .^ 2 * (weights[i] * volume_jacobian_)
            linf_error = @. max(linf_error, abs(diff))
        end
    end

    # For L2 error, divide by total volume
    total_volume_ = total_volume(mesh)
    l2_error = @. sqrt(l2_error / total_volume_)

    return l2_error, linf_error
end
