function max_dt(u::CuArray, t, mesh::TreeMesh{1},
                constant_speed::True, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    # FIXME: The is a temporary workaround to avoid scalar indexing issue.
    inverse_jacobian = Array(cache.elements.inverse_jacobian)

    for element in eachelement(dg, cache)
        max_lambda1, = max_abs_speeds(equations)
        inv_jacobian = inverse_jacobian[element]
        max_scaled_speed = max(max_scaled_speed, inv_jacobian * max_lambda1)
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end

function max_dt(u::CuArray, t, mesh::TreeMesh{1},
                constant_speed::False, equations, dg::DG, cache)
    # to avoid a division by zero if the speed vanishes everywhere,
    # e.g. for steady-state linear advection
    max_scaled_speed = nextfloat(zero(t))

    # FIXME: This is a temporary workaround to avoid scalar indexing issue.
    u = Array(u)
    inverse_jacobian = Array(cache.elements.inverse_jacobian)

    for element in eachelement(dg, cache)
        max_lambda1 = zero(max_scaled_speed)
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            lambda1, = max_abs_speeds(u_node, equations)
            max_lambda1 = max(max_lambda1, lambda1)
        end
        inv_jacobian = inverse_jacobian[element]
        max_scaled_speed = max(max_scaled_speed, inv_jacobian * max_lambda1)
    end

    return 2 / (nnodes(dg) * max_scaled_speed)
end
