# Everything specific about semidiscretization hyperbolic for PDE solvers.

mutable struct SemidiscretizationHyperbolicGPU{Mesh, Equations, InitialCondition,
                                               BoundaryConditions,
                                               SourceTerms, Solver, Cache} <:
               AbstractSemidiscretization
    # TODO: Need to further discuss if const could apply to mesh and performance counter.
    mesh::Mesh
    equations::Equations

    const initial_condition::InitialCondition
    const boundary_conditions::BoundaryConditions
    const source_terms::SourceTerms
    const solver::Solver
    const cache::Cache
    performance_counter::PerformanceCounter

    function SemidiscretizationHyperbolicGPU{Mesh, Equations, InitialCondition,
                                             BoundaryConditions, SourceTerms, Solver,
                                             Cache}(mesh::Mesh, equations::Equations,
                                                    initial_condition::InitialCondition,
                                                    boundary_conditions::BoundaryConditions,
                                                    source_terms::SourceTerms,
                                                    solver::Solver,
                                                    cache::Cache) where {Mesh, Equations,
                                                                         InitialCondition,
                                                                         BoundaryConditions,
                                                                         SourceTerms,
                                                                         Solver,
                                                                         Cache}
        performance_counter = PerformanceCounter()

        new(mesh, equations, initial_condition, boundary_conditions, source_terms,
            solver, cache, performance_counter)
    end
end

function SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver;
                                         boundary_conditions,
                                         source_terms = nothing,
                                         # `RealT` is used as real type for node locations etc.
                                         # `uEltype` is used as element type of solutions etc.
                                         RealT = real(solver),
                                         uEltype = RealT)
    @assert ndims(mesh) == ndims(equations)

    cache = create_cache_gpu(mesh, equations, solver, RealT, uEltype)

    _boundary_conditions = digest_boundary_conditions(boundary_conditions, mesh, solver, cache)
    check_periodicity_mesh_boundary_conditions(mesh, _boundary_conditions)

    SemidiscretizationHyperbolicGPU{typeof(mesh), typeof(equations),
                                    typeof(initial_condition), typeof(_boundary_conditions),
                                    typeof(source_terms), typeof(solver),
                                    typeof(cache)}(mesh, equations, initial_condition,
                                                   _boundary_conditions, source_terms,
                                                   solver, cache)
end

@inline Base.ndims(semi::SemidiscretizationHyperbolicGPU) = ndims(semi.mesh)
@inline Base.real(semi::SemidiscretizationHyperbolicGPU) = real(semi.solver)

@inline function mesh_equations_solver_cache(semi::SemidiscretizationHyperbolicGPU)
    (; mesh, equations, solver, cache) = semi
    return mesh, equations, solver, cache
end

function calc_error_norms(func, u_ode, t, analyzer, semi::SemidiscretizationHyperbolicGPU,
                          cache_analysis)
    (; mesh, equations, initial_condition, solver, cache) = semi
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition, solver,
                     cache, cache_analysis)
end

function compute_coefficients_gpu(t, semi::SemidiscretizationHyperbolicGPU)
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

function print_boundary_conditions(io, semi::SemiHypMeshBCSolver{<:Union{TreeMesh,
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
