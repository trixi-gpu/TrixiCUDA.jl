include("basis_lobatto_legendre.jl")
include("cache.jl")
include("common.jl")
include("containers_1d.jl")
include("containers_2d.jl")
include("containers_3d.jl")
include("indicators.jl")
include("indicators_1d.jl")
include("indicators_2d.jl")
include("indicators_3d.jl")
include("dg_1d.jl")
include("dg_2d.jl")
include("dg_3d.jl")
include("dgsem_gpu.jl")
include("dg.jl")

function rhs_hyperbolic!(du_ode, u_ode, semi::SemidiscretizationHyperbolicGPU, t)

    (; mesh, equations, boundary_conditions, source_terms, solver, cache) = semi
    rhs_hyperbolic!(du, u, t, mesh, equations, boundary_conditions, source_terms, 
                    solver, cache)
    return nothing
end

function semidiscretizeGPU(semi::SemidiscretizationHyperbolicGPU, tspan;
                           jac_prototype::Union{AbstractMatrix, Nothing} = nothing,
                           colorvec::Union{AbstractVector, Nothing} = nothing)
    # Computing coefficients on GPUs may not be as fast as on CPUs due to the overhead. 
    # Therefore, we currently use the GPU version. Note that the actual speedup on GPUs
    # largely depends on the problem size (e.g., large arrays typically gain much more 
    # speedup than small arrays). 

    # TODO: Discuss how performance can be improved by initializing the coefficients on 
    # the CPU and then transferring them to the GPU. 

    tspan = convert_tspan(semi, tspan)
    u0_ode = compute_coefficients_gpu(first(tspan), semi)

    iip = true
    specialize = FullSpecialize
    if jac_prototype !== nothing
        ode = ODEFunction(rhs_hyperbolic!;
                          jac_prototype = convert.(eltype(u0_ode), jac_prototype),
                          colorvec = colorvec)
        return ODEProblem{iip, specialize}(ode, u0_ode, tspan, semi)
    end
    return ODEProblem{iip, specialize}(rhs_hyperbolic!, u0_ode, tspan, semi)
end
