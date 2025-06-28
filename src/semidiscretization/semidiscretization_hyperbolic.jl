# Everything specific about semidiscretization hyperbolic for PDE solvers.

# Similar to `SemidiscretizationHyperbolic` in Trixi.jl 
# No need to adapt the `SemidiscretizationHyperbolic` struct as it is already GPU compatible
mutable struct SemidiscretizationHyperbolicGPU{Mesh, Equations, InitialCondition,
                                               BoundaryConditions,
                                               SourceTerms, Solver, CacheGPU, CacheCPU} <:
               AbstractSemidiscretization
    mesh::Mesh
    equations::Equations

    # This guy is a bit messy since we abuse it as some kind of "exact solution"
    # although this doesn't really exist...
    initial_condition::InitialCondition

    boundary_conditions::BoundaryConditions
    source_terms::SourceTerms
    solver::Solver
    cache_gpu::CacheGPU
    cache_cpu::CacheCPU
    performance_counter::PerformanceCounter

    function SemidiscretizationHyperbolicGPU{Mesh, Equations, InitialCondition,
                                             BoundaryConditions, SourceTerms, Solver,
                                             CacheGPU, CacheCPU}(mesh::Mesh, equations::Equations,
                                                                 initial_condition::InitialCondition,
                                                                 boundary_conditions::BoundaryConditions,
                                                                 source_terms::SourceTerms,
                                                                 solver::Solver,
                                                                 cache_gpu::CacheGPU,
                                                                 cache_cpu::CacheCPU) where {Mesh, Equations,
                                                                                             InitialCondition,
                                                                                             BoundaryConditions,
                                                                                             SourceTerms,
                                                                                             Solver,
                                                                                             CacheGPU, CacheCPU}
        performance_counter = PerformanceCounter()

        new(mesh, equations, initial_condition, boundary_conditions, source_terms,
            solver, cache_gpu, cache_cpu, performance_counter)
    end
end

"""
    SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver;
                                    source_terms=nothing,
                                    boundary_conditions=boundary_condition_periodic,
                                    RealT=real(solver),
                                    uEltype=RealT,
                                    initial_cache_gpu=NamedTuple(),
                                    initial_cache_cpu=NamedTuple())

Construct a semidiscretization of a hyperbolic PDE for GPU-accelerated computations. This version 
stores and manages computational caches on both GPU and CPU to achieve efficient data transfer.

!!! warning "Experimental implementation"
    This is an experimental implementation and may change or be removed in future releases due to 
    ongoing performance optimizations.
"""
function SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver;
                                         source_terms = nothing,
                                         boundary_conditions = boundary_condition_periodic,
                                         RealT = real(solver), # `RealT` is used as real type for node locations etc.
                                         uEltype = RealT, # `uEltype` is used as element type of solutions etc.
                                         initial_cache_gpu = NamedTuple(),
                                         initial_cache_cpu = NamedTuple())
    @assert ndims(mesh) == ndims(equations)

    cache_gpu = (; create_cache_gpu(mesh, equations, solver, RealT, uEltype)...,
                 initial_cache_gpu...)
    cache_cpu = (; create_cache(mesh, equations, solver, RealT, uEltype)...,
                 initial_cache_cpu...)

    _boundary_conditions = digest_boundary_conditions(boundary_conditions, mesh, solver,
                                                      cache_cpu)

    check_periodicity_mesh_boundary_conditions(mesh, _boundary_conditions)

    SemidiscretizationHyperbolicGPU{typeof(mesh), typeof(equations),
                                    typeof(initial_condition), typeof(_boundary_conditions),
                                    typeof(source_terms), typeof(solver),
                                    typeof(cache_gpu), typeof(cache_cpu)}(mesh, equations,
                                                                          initial_condition,
                                                                          _boundary_conditions,
                                                                          source_terms, solver,
                                                                          cache_gpu, cache_cpu)
end

@inline Base.ndims(semi::SemidiscretizationHyperbolicGPU) = ndims(semi.mesh)

@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolicGPU)
    # Unpack CPU cache or GPU cache?
    (; mesh, equations, solver, cache_gpu) = semi
    return mesh, equations, solver, cache_gpu
end

function calc_error_norms(func, u_ode, t, analyzer, semi::SemidiscretizationHyperbolicGPU,
                          cache_analysis)
    # Unpack CPU cache or GPU cache?
    (; mesh, equations, initial_condition, solver, cache_gpu) = semi
    u = wrap_array(u_ode, mesh, equations, solver, cache_gpu)

    calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver,
                     cache_gpu, cache_analysis)
end

# Similar to `compute_coefficients` in Trixi.jl but calls GPU kernel
function compute_coefficients_gpu(t, semi::SemidiscretizationHyperbolicGPU)

    # Call `compute_coefficients_gpu` defined in `src/semidiscretization/semidiscretization.jl`
    compute_coefficients_gpu(semi.initial_condition, t, semi)
end

function Base.show(io::IO, ::MIME"text/plain", semi::SemidiscretizationHyperbolicGPU)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationHyperbolicGPU")
        summary_line(io, "#spatial dimensions", ndims(semi.equations))
        summary_line(io, "mesh", semi.mesh)
        summary_line(io, "equations", semi.equations |> typeof |> nameof)
        summary_line(io, "initial condition", semi.initial_condition)

        print_boundary_conditions(io, semi)

        summary_line(io, "source terms", semi.source_terms)
        summary_line(io, "solver", semi.solver |> typeof |> nameof)
        summary_line(io, "total #DOFs per field", ndofs(semi))
        summary_footer(io)
    end
end

# type alias for dispatch in printing of boundary conditions
#! format: off
const SemiHypMeshBCSolver{Mesh, BoundaryConditions, Solver} =
        SemidiscretizationHyperbolicGPU{Mesh,
                                        Equations,
                                        InitialCondition,
                                        BoundaryConditions,
                                        SourceTerms,
                                        Solver} where {Equations,
                                                    InitialCondition,
                                                    SourceTerms}
#! format: on

# generic fallback: print the type of semi.boundary_condition.
function print_boundary_conditions(io, semi::SemiHypMeshBCSolver)
    summary_line(io, "boundary conditions", typeof(semi.boundary_conditions))
end

function print_boundary_conditions(io,
                                   semi::SemiHypMeshBCSolver{<:Any,
                                                             <:UnstructuredSortedBoundaryTypes})
    (; boundary_conditions) = semi
    (; boundary_dictionary) = boundary_conditions
    summary_line(io, "boundary conditions", length(boundary_dictionary))
    for (boundary_name, boundary_condition) in boundary_dictionary
        summary_line(increment_indent(io), boundary_name, typeof(boundary_condition))
    end
end

function print_boundary_conditions(io, semi::SemiHypMeshBCSolver{<:Any, <:NamedTuple})
    (; boundary_conditions) = semi
    summary_line(io, "boundary conditions", length(boundary_conditions))
    bc_names = keys(boundary_conditions)
    for (i, bc_name) in enumerate(bc_names)
        summary_line(increment_indent(io), String(bc_name),
                     typeof(boundary_conditions[i]))
    end
end

function print_boundary_conditions(io,
                                   semi::SemiHypMeshBCSolver{<:Union{TreeMesh,
                                                                     StructuredMesh},
                                                             <:Union{Tuple, NamedTuple,
                                                                     AbstractArray}})
    summary_line(io, "boundary conditions", 2 * ndims(semi))
    bcs = semi.boundary_conditions

    summary_line(increment_indent(io), "negative x", bcs[1])
    summary_line(increment_indent(io), "positive x", bcs[2])
    if ndims(semi) > 1
        summary_line(increment_indent(io), "negative y", bcs[3])
        summary_line(increment_indent(io), "positive y", bcs[4])
    end
    if ndims(semi) > 2
        summary_line(increment_indent(io), "negative z", bcs[5])
        summary_line(increment_indent(io), "positive z", bcs[6])
    end
end
