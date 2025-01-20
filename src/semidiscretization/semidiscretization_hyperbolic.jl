# Everything specific about semidiscretization hyperbolic for PDE solvers.

# Similar to `SemidiscretizationHyperbolic` in Trixi.jl 
# No need to adapt the `SemidiscretizationHyperbolic` struct as it is already GPU compatible

# Outer constructor for GPU type
function SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver;
                                         source_terms = nothing,
                                         boundary_conditions = boundary_condition_periodic,
                                         RealT = real(solver), # `RealT` is used as real type for node locations etc.
                                         uEltype = RealT, # `uEltype` is used as element type of solutions etc.
                                         initial_cache = NamedTuple())
    @assert ndims(mesh) == ndims(equations)

    cache = (; create_cache_gpu(mesh, equations, solver, RealT, uEltype)...,
             initial_cache...)
    _boundary_conditions = digest_boundary_conditions(boundary_conditions, mesh, solver,
                                                      cache)

    check_periodicity_mesh_boundary_conditions(mesh, _boundary_conditions)

    # Return the CPU type (GPU compatible)
    SemidiscretizationHyperbolic{typeof(mesh), typeof(equations),
                                 typeof(initial_condition),
                                 typeof(_boundary_conditions), typeof(source_terms),
                                 typeof(solver), typeof(cache)}(mesh, equations,
                                                                initial_condition,
                                                                _boundary_conditions,
                                                                source_terms, solver,
                                                                cache)
end

# Similar to `compute_coefficients` in Trixi.jl but calls GPU kernel
function compute_coefficients_gpu(t, semi::SemidiscretizationHyperbolic)

    # Call `compute_coefficients_gpu` defined in `src/semidiscretization/semidiscretization.jl`
    compute_coefficients_gpu(semi.initial_condition, t, semi)
end
