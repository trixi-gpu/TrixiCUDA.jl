include("cache.jl")
include("common.jl")
include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
include("dg_1d.jl")
include("dg_2d.jl")
include("dg_3d.jl")
include("dg.jl")

# See also `rhs!` function in Trixi.jl
function rhs_gpu!(du_ode, u_ode, semi::SemidiscretizationHyperbolic, t)
    (; mesh, equations, boundary_conditions, source_terms, solver, cache) = semi

    # In Trixi.jl, function `wrap_array` is called to adapt adaptive mesh refinement (AMR).
    # We are currently not considering AMR in TrixiCUDA.jl, so this step is not needed here. 
    # For more details, see https://trixi-framework.github.io/Trixi.jl/stable/conventions/#Array-types-and-wrapping
    # TODO: Adapt `wrap_array` on GPUs for AMR
    # u = wrap_array(u_ode, mesh, equations, solver, cache)
    # du = wrap_array(du_ode, mesh, equations, solver, cache)

    rhs_gpu!(du_ode, u_ode, t, mesh, equations, boundary_conditions, source_terms, solver, cache)

    return nothing
end

# See also `semidiscretize` function in Trixi.jl
function semidiscretizeGPU(semi::SemidiscretizationHyperbolic, tspan)
    u0_ode = compute_coefficients_gpu(first(tspan), semi)

    iip = true
    specialize = FullSpecialize
    return ODEProblem{iip, specialize}(rhs_gpu!, u0_ode, tspan, semi)
end
