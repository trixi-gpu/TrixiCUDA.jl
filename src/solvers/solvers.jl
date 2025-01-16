include("cache.jl")
include("common.jl")
include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
include("dg_1d.jl")
include("dg_2d.jl")
include("dg_3d.jl")
include("dg.jl")
include("dgsem.jl")

# See also `rhs!` function in Trixi.jl
function rhs_gpu!(du_ode, u_ode, semi::SemidiscretizationHyperbolic, t)
    (; mesh, equations, boundary_conditions, source_terms, solver, cache) = semi

    # In Trixi.jl, function `wrap_array` is called to adapt adaptive mesh refinement (AMR).
    # For more details, see https://trixi-framework.github.io/Trixi.jl/stable/conventions/#Array-types-and-wrapping
    u = wrap_array(u_ode, mesh, equations, solver, cache)
    du = wrap_array(du_ode, mesh, equations, solver, cache)

    rhs_gpu!(du, u, t, mesh, equations, boundary_conditions, source_terms, solver, cache)

    return nothing
end

# See also `semidiscretize` function in Trixi.jl
function semidiscretizeGPU(semi::SemidiscretizationHyperbolic, tspan)
    # Computing coefficients on GPUs may not be as fast as on CPUs due to the overhead. 
    # Therefore, we currently use the GPU version. Note that the actual speedup on GPUs
    # largely depends on the problem size (e.g., large arrays typically gain much more 
    # speedup than small arrays). 

    u0_ode = compute_coefficients_gpu(first(tspan), semi)
    # TODO: We may switch back to CPUs in the future
    # u0_ode = CuArray(compute_coefficients(first(tspan), semi))

    iip = true
    specialize = FullSpecialize
    return ODEProblem{iip, specialize}(rhs_gpu!, u0_ode, tspan, semi)
end
