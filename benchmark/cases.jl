using Trixi, TrixiShallowWater
using StaticArrays: SVector

# Include only configuration factories at load time; allocate each problem when measured.
const RHS_CASES = [(; name = string(splitext(basename(file))[1], "_", d, "d"), build = include(file))
                   for d in 1:3
                   for file in filter(endswith(".jl"), readdir(joinpath(@__DIR__, "cases", "cases_$(d)d"); join = true))]

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

initial_input(semi, t) = Array(TrixiCUDA.compute_coefficients_gpu(t, semi))
