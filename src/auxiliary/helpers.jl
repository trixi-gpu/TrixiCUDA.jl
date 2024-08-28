# Some helper functions and function extensions from Trixi.jl.

# Ref: `get_node_vars(u, equations, solver::DG, indices...)` in Trixi.jl
@inline function get_node_vars(u, equations, indices...)
    return SVector(ntuple(@inline(v->u[v, indices...]),
                          Val(nvariables(equations))))
end

# Ref: `get_node_coords(x, equations, solver::DG, indices...)` in Trixi.jl
@inline function get_node_coords(x, equations, indices...)
    return SVector(ntuple(@inline(idx->x[idx, indices...]),
                          Val(ndims(equations))))
end

# Ref: `get_surface_node_vars(u, equations, solver::DG, indices...)` in Trixi.jl
@inline function get_surface_node_vars(u, equations, indices...)
    u_ll = SVector(ntuple(@inline(v->u[1, v, indices...]),
                          Val(nvariables(equations))))
    u_rr = SVector(ntuple(@inline(v->u[2, v, indices...]),
                          Val(nvariables(equations))))

    return u_ll, u_rr
end

# Helper function for checking `cache.mortars`
@inline function check_cache_mortars(cache)
    if iszero(length(cache.mortars.orientations))
        return True()
    else
        return False()
    end
end

# Callable function to replace the `boundary_condition_periodic` from Trixi.jl
@inline function boundary_condition_periodic_callable(u_inner, orientation,
                                                      direction, x, t, surface_flux, equations)
    return nothing
end

# Replace the `boundary_condition_periodic` from Trixi.jl with a callable one
function replace_boundary_conditions(boundary_conditions::NamedTuple)
    keys_ = keys(boundary_conditions)
    values_ = (func == boundary_condition_periodic ? boundary_condition_periodic_callable : func
               for func in values(boundary_conditions))
    return NamedTuple{keys_}(values_)
end
