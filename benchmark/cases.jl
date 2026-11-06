using Trixi, TrixiShallowWater
using StaticArrays: SVector

rhs_case_files() = filter(file -> endswith(file, ".jl"), readdir(joinpath(@__DIR__, "cases"); join = true))

# Include only configuration factories at load time; allocate each problem when measured.
const RHS_CASES = [(; name = splitext(basename(file))[1], build = include(file)) for file in rhs_case_files()]
rhs_cases() = RHS_CASES

make_problem(case; kwargs...) = rhs_problem(; case.build(; kwargs...)...)

function rhs_problem(; equations, initial_condition, coordinates_min, coordinates_max,
                     initial_refinement_level, periodicity, polydeg, surface_flux,
                     volume_integral, boundary_conditions, source_terms,
                     refinement_patches = ())
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level, refinement_patches, periodicity, RealT = Float64)
    solver = DGSEMGPU(; polydeg, surface_flux, volume_integral, RealT = Float64)
    return SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver; boundary_conditions, source_terms)
end

function initial_input(semi, t)
    (; mesh, equations, initial_condition, solver, cache_cpu) = semi
    nn = Trixi.nnodes(solver)
    ne = Trixi.nelements(solver, cache_cpu)
    nodes = CartesianIndices(ntuple(_ -> nn, ndims(equations)))
    u = Array{Float64}(undef, Trixi.nvariables(equations), size(nodes)..., ne)
    # The shallow-water 2D EC example also needs the element ID at shared nodes.
    elementwise = applicable(initial_condition, zero(SVector{ndims(equations), Float64}), t, 1, equations)
    if !elementwise
        # Preserve Trixi's one-sided sampling of discontinuous 1D initial data.
        Trixi.compute_coefficients!(nothing, u, initial_condition, t, mesh, equations, solver, cache_cpu)
        return u
    end
    for element in Trixi.eachelement(solver, cache_cpu), node in nodes
        indices = Tuple(node)
        x = Trixi.get_node_coords(cache_cpu.elements.node_coordinates, equations, solver, indices..., element)
        state = initial_condition(x, t, element, equations)
        Trixi.set_node_vars!(u, state, equations, solver, indices..., element)
    end
    return u
end
