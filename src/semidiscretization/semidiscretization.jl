include("semidiscretization_hyperbolic.jl")

# Similar to `compute_coefficients` in Trixi.jl but calls GPU kernel
function compute_coefficients_gpu(func, t, semi::AbstractSemidiscretization)
    u_ode = allocate_coefficients(mesh_equations_solver_cache(semi)...)

    # Call `compute_coefficients_gpu` defined below
    u_ode = compute_coefficients_gpu(u_ode, func, t, semi)
    return u_ode
end

# Compute the coefficients for 1D problems on the GPU
function compute_coefficients_gpu(u_ode, func, t, semi::AbstractSemidiscretization)
    u = wrap_array(u_ode, semi)
    u = CuArray(u)

    # Call `compute_coefficients_gpu` defined in `src/solvers/dg.jl`
    u_computed = compute_coefficients_gpu(u, func, t, mesh_equations_solver_cache(semi)...)
    return u_computed
end

# function integrate(func::Func, u_ode::Array, semi::AbstractSemidiscretization;
#                    normalize = true) where {Func}
#     mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

#     # u = wrap_array(u_ode, mesh, equations, solver, cache)
#     integrate(func, u_ode, mesh, equations, solver, cache, normalize = normalize)
# end
