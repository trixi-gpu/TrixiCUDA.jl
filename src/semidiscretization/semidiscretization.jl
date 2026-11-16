include("semidiscretization_hyperbolic.jl")

# Similar to `compute_coefficients` in Trixi.jl but calls GPU kernels. Unlike Trixi.jl, which
# allocates a flat vector and wraps it, allocate the GPU array directly in the
# `(nvariables, nnodes, ..., nnodes, nelements)` layout used by the kernels.
function compute_coefficients_gpu(func, t, semi::AbstractSemidiscretization)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = CuArray{eltype(cache.elements)}(undef, nvariables(equations),
                                        ntuple(_ -> nnodes(solver), ndims(mesh))...,
                                        nelements(solver, cache))

    # Call `compute_coefficients_gpu` defined in `src/solvers/dg.jl`
    return compute_coefficients_gpu(u, func, t, mesh, equations, solver, cache)
end
