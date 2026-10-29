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

# See also `rhs_hyperbolic!` in Trixi.jl.
function rhs_hyperbolic!(du_ode, u_ode, semi::SemidiscretizationHyperbolicGPU, t)
    (; mesh, equations, boundary_conditions, source_terms, solver,
     cache_gpu, cache_cpu) = semi

    # In Trixi.jl, function `wrap_array` is called to adapt adaptive mesh refinement (AMR).
    # For more details, see https://trixi-framework.github.io/Trixi.jl/stable/conventions/#Array-types-and-wrapping
    u = wrap_array(u_ode, mesh, equations, solver, cache_gpu)
    du = wrap_array(du_ode, mesh, equations, solver, cache_gpu)
    backend = trixi_backend(u)

    rhs_hyperbolic!(backend, du, u, t, mesh, equations, boundary_conditions, source_terms, solver, cache_gpu, cache_cpu)

    return nothing
end

"""
    semidiscretizeGPU(semi::SemidiscretizationHyperbolicGPU, tspan;
                     jac_prototype=nothing, colorvec=nothing)

Wrap the GPU-accelerated semidiscretization `semi` as an ODE problem over the time interval `tspan`,
which can be solved using the [SciML ecosystem](https://diffeq.sciml.ai/latest/).
As in Trixi's `semidiscretize`, `tspan` is converted to `real(semi)` before evaluating the
initial condition. Optional `jac_prototype` and `colorvec` are stored in the ODE function;
the Jacobian prototype's element type is converted to that of the initial state.
Its storage backend is preserved, so callers must choose one supported by their ODE solver.
These options do not add GPU automatic differentiation or implicit-solver support.

!!! warning "Experimental implementation"
    This is an experimental feature and may change or be removed in future releases due to 
    ongoing performance optimizations.
"""
function semidiscretizeGPU(semi::SemidiscretizationHyperbolicGPU, tspan;
                           jac_prototype::Union{AbstractMatrix, Nothing} = nothing,
                           colorvec::Union{AbstractVector, Nothing} = nothing)
    # Computing coefficients on GPUs may not be as fast as on CPUs due to the overhead. 
    # Therefore, we currently use the GPU version. Note that the actual speedup on GPUs
    # largely depends on the problem size (e.g., large arrays typically gain much more 
    # speedup than small arrays). 

    tspan = convert_tspan(semi, tspan)
    u0_ode = compute_coefficients_gpu(first(tspan), semi)
    # TODO: We may switch back to CPUs in the future
    # u0_ode = CuArray(compute_coefficients(first(tspan), semi))

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
