using Trixi, TrixiShallowWater
using StaticArrays: SVector

# Include only configuration factories at load time; allocate each problem when measured.
const RHS_CASES = [(; name = splitext(basename(file))[1], build = include(file))
                   for file in filter(endswith(".jl"), readdir(joinpath(@__DIR__, "cases"); join = true))]

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
    (; mesh, equations, initial_condition, solver) = semi
    # Build the element data on the host with Trixi, independent of the cache layout
    # of the revision being measured, so both revisions get the same input.
    basis = Trixi.LobattoLegendreBasis(Float64, Trixi.polydeg(solver))
    elements = Trixi.init_elements(Trixi.local_leaf_cells(mesh.tree), mesh, equations, basis, Float64, Float64)
    cache = (; elements)
    nn = Trixi.nnodes(solver)
    ne = Trixi.nelements(solver, cache)
    nodes = CartesianIndices(ntuple(_ -> nn, ndims(equations)))
    u = Array{Float64}(undef, Trixi.nvariables(equations), size(nodes)..., ne)
    elementwise = applicable(initial_condition, zero(SVector{ndims(equations), Float64}), t, 1, equations)
    # Use Trixi's standard initialization for initial conditions that don't require
    # an element ID. This preserves one-sided sampling in 1D.
    if !elementwise
        Trixi.compute_coefficients!(nothing, u, initial_condition, t, mesh, equations, solver, cache)
        return u
    end
    # Pass the element ID to initial conditions that need it, such as the 
    # discontinuous bottom topography in the 2D shallow-water EC case.
    for element in Trixi.eachelement(solver, cache), node in nodes
        indices = Tuple(node)
        x = Trixi.get_node_coords(cache.elements.node_coordinates, equations, solver, indices..., element)
        state = initial_condition(x, t, element, equations)
        Trixi.set_node_vars!(u, state, equations, solver, indices..., element)
    end
    return u
end
