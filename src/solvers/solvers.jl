include("cache.jl")
include("common.jl")
include("dg_1d.jl")
include("dg_2d.jl")
include("dg_3d.jl")

# Ref: `rhs!` function in Trixi.jl
function rhs_gpu!(du_ode, u_ode, semi::SemidiscretizationHyperbolic, t)
    (; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi

    u = wrap_array(u_ode, mesh, equations, solver, cache)
    du = wrap_array(du_ode, mesh, equations, solver, cache)

    rhs_gpu!(du, u, t, mesh, equations, boundary_conditions, source_terms, solver, cache)

    return nothing
end

# Ref: `semidiscretize` function in Trixi.jl
function semidiscretize_gpu(semi::SemidiscretizationHyperbolic, tspan)
    u0_ode = compute_coefficients(first(tspan), semi)

    iip = true
    specialize = FullSpecialize
    return ODEProblem{iip, specialize}(rhs_gpu!, u0_ode, tspan, semi)
end
