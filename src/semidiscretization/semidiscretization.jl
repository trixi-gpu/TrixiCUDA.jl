include("semidiscretization_hyperbolic.jl")

# Allocate the GPU array directly in DG layout used by the kernels.
function compute_coefficients_gpu(func, t, semi::AbstractSemidiscretization)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = CuArray{eltype(cache.elements)}(undef, nvariables(equations),
                                        ntuple(_ -> nnodes(solver), ndims(mesh))...,
                                        nelements(solver, cache))

    return compute_coefficients_gpu(u, func, t, mesh, equations, solver, cache)
end
