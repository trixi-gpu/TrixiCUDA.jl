using Trixi, TrixiShallowWater
using StaticArrays: SVector
using StableRNGs: StableRNG
using Random: rand

rhs_case_files() = filter(file -> endswith(file, ".jl"), readdir(joinpath(@__DIR__, "cases"); join = true))

# Include only configuration factories at load time; allocate each problem when measured.
const RHS_CASES = [(; name = splitext(basename(file))[1], build = include(file)) for file in rhs_case_files()]
rhs_cases() = RHS_CASES

make_problem(case) = rhs_problem(; case.build()...)

function rhs_problem(; equations, coordinates_min, coordinates_max,
                     initial_refinement_level, periodicity, polydeg, surface_flux,
                     volume_integral, boundary_conditions, source_terms,
                     refinement_patches = ())
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level, refinement_patches, periodicity, RealT = Float64)
    solver = DGSEMGPU(; polydeg, surface_flux, volume_integral, RealT = Float64)
    # No initial condition or ODE solve is evaluated; u is supplied directly below.
    return SemidiscretizationHyperbolicGPU(mesh, equations, nothing, solver; boundary_conditions, source_terms)
end

function random_state(rng, equations)
    values = rand(rng, Float64, Trixi.nvariables(equations))
    is_euler = equations isa Union{CompressibleEulerEquations1D, CompressibleEulerEquations2D,
                                   CompressibleEulerEquations3D}
    is_mhd = equations isa Union{IdealGlmMhdEquations1D, IdealGlmMhdEquations2D,
                                 IdealGlmMhdEquations3D}
    if is_euler || is_mhd
        # Random primitive variables avoid negative density/pressure in random
        # conservative vectors. MHD always has three velocity components.
        values[1] += 1.0
        values[is_euler ? ndims(equations) + 2 : 5] += 1.0
        return prim2cons(SVector{length(values)}(values), equations)
    elseif equations isa Union{ShallowWaterEquations1D, ShallowWaterEquations2D}
        # Shallow-water primitives use total height H = h + bottom topography.
        values[1] += 1.0 + values[end]
        return prim2cons(SVector{length(values)}(values), equations)
    end
    return SVector{length(values)}(values)
end

function random_input(equations, solver, cache, seed)
    nv = Trixi.nvariables(equations)
    nn = Trixi.nnodes(solver)
    ne = Trixi.nelements(solver, cache)
    u = Array{Float64}(undef, nv, ntuple(_ -> nn, ndims(equations))..., ne)
    nodes = reshape(u, nv, :)
    rng = StableRNG(seed)
    for j in axes(nodes, 2)
        nodes[:, j] = random_state(rng, equations)
    end
    return u
end
