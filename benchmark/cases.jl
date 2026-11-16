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
    (; equations, initial_condition, solver, cache) = semi
    elementwise = applicable(initial_condition, zero(SVector{ndims(equations), Float64}),
                             t, 1, equations)

    # Use Trixi's standard initialization for initial conditions that don't require
    # an element ID. This preserves one-sided sampling in 1D.
    elementwise || return Array(TrixiCUDA.compute_coefficients_gpu(t, semi))

    # Pass the element ID to initial conditions that need it, such as the 
    # discontinuous bottom topography in the 2D shallow-water EC case.
    node_coordinates = Array(cache.elements.node_coordinates)
    nodes = CartesianIndices(ntuple(_ -> Trixi.nnodes(solver), ndims(equations)))
    u = Array{eltype(cache.elements)}(undef, Trixi.nvariables(equations), size(nodes)...,
                                      Trixi.nelements(solver, cache))
    for element in Trixi.eachelement(solver, cache), node in nodes
        x = Trixi.get_node_coords(node_coordinates, equations, solver, Tuple(node)..., element)
        state = initial_condition(x, t, element, equations)
        Trixi.set_node_vars!(u, state, equations, solver, Tuple(node)..., element)
    end
    return u
end
