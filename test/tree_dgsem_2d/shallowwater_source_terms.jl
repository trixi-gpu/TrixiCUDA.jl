module TestShallowWaterSourceTerms2D

using Trixi, TrixiShallowWater, TrixiCUDA
using Test
using CUDA

# Adapted from TrixiShallowWater.jl (66644da):
# https://github.com/trixi-framework/TrixiShallowWater.jl/blob/66644da94ebd3ad881f17379e17ffea456e15e51/examples/tree_2d_dgsem/elixir_shallowwater_source_terms.jl
include("../test_macros.jl")

@testset "Shallow Water Source Terms 2D" begin
    equations = ShallowWaterEquations2D(gravity = 9.81)

    initial_condition = initial_condition_convergence_test # MMS EOC test

    surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_fjordholm_etal)
    volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
    solver = DGSEM(polydeg = 3,
                   surface_flux = surface_flux,
                   volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
    solver_gpu = DGSEMGPU(polydeg = 3,
                          surface_flux = surface_flux,
                          volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (0.0, 0.0)
    coordinates_max = (sqrt(2.0), sqrt(2.0))
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 3,
                    periodicity = true)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        source_terms = source_terms_convergence_test,
                                        boundary_conditions = boundary_condition_periodic)
    semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver_gpu,
                                               source_terms = source_terms_convergence_test,
                                               boundary_conditions = boundary_condition_periodic)

    tspan = tspan_gpu = (0.0, 0.1)
    t = t_gpu = 0.1

    # Semi on CPU
    (; mesh, equations, boundary_conditions, source_terms, solver, cache) = semi

    # Semi on GPU
    equations_gpu, mesh_gpu, solver_gpu = semi_gpu.equations, semi_gpu.mesh, semi_gpu.solver
    cache_gpu, cache_cpu = semi_gpu.cache_gpu, semi_gpu.cache_cpu
    boundary_conditions_gpu, source_terms_gpu = semi_gpu.boundary_conditions, semi_gpu.source_terms

    # ODE on CPU
    ode = semidiscretize(semi, tspan)
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
    du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

    # ODE on GPU
    ode_gpu = semidiscretizeGPU(semi_gpu, tspan_gpu)
    u_gpu_ = copy(ode_gpu.u0)
    du_gpu_ = similar(u_gpu_)
    u_gpu = TrixiCUDA.wrap_array(u_gpu_, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
    du_gpu = TrixiCUDA.wrap_array(du_gpu_, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)

    # Tests for semidiscretization process
    @test_approx (u_gpu, u)
    Trixi.set_zero!(du, solver, cache)

    TrixiCUDA.calc_volume_integral!(CUDA.CUDABackend(), du_gpu, u_gpu, mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu), equations_gpu,
                                    solver_gpu.volume_integral, solver_gpu, cache_gpu, cache_cpu)
    Trixi.calc_volume_integral!(nothing, du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                equations, solver.volume_integral, solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.prolong2interfaces!(CUDA.CUDABackend(), cache_gpu, u_gpu, mesh_gpu, equations_gpu, solver_gpu)
    Trixi.prolong2interfaces!(nothing, cache, u, mesh, equations, solver)
    @test_approx (cache_gpu.interfaces.u, cache.interfaces.u)

    TrixiCUDA.calc_interface_flux!(CUDA.CUDABackend(), cache_gpu.elements.surface_flux_values, mesh_gpu,
                                   Trixi.have_nonconservative_terms(equations_gpu), equations_gpu, solver_gpu.surface_integral, solver_gpu, cache_gpu)
    Trixi.calc_interface_flux!(nothing, cache.elements.surface_flux_values, mesh,
                               Trixi.have_nonconservative_terms(equations), equations,
                               solver.surface_integral, solver, cache)
    @test_approx (cache_gpu.elements.surface_flux_values,
                  cache.elements.surface_flux_values)

    TrixiCUDA.prolong2boundaries!(CUDA.CUDABackend(), cache_gpu, u_gpu, mesh_gpu, equations_gpu, solver_gpu)
    Trixi.prolong2boundaries!(nothing, cache, u, mesh, equations, solver)
    @test_approx (cache_gpu.boundaries.u, cache.boundaries.u)

    TrixiCUDA.calc_boundary_flux!(CUDA.CUDABackend(), cache_gpu, t_gpu, boundary_conditions_gpu, mesh_gpu, equations_gpu, solver_gpu.surface_integral,
                                  solver_gpu)
    Trixi.calc_boundary_flux!(nothing, cache, t, boundary_conditions, mesh, equations,
                              solver.surface_integral, solver)
    @test_approx (cache_gpu.elements.surface_flux_values,
                  cache.elements.surface_flux_values)

    TrixiCUDA.prolong2mortars!(cache_gpu, u_gpu, mesh_gpu, equations_gpu, solver_gpu.mortar, solver_gpu)
    Trixi.prolong2mortars!(cache, u, mesh, equations, solver.mortar, solver)
    @test_approx (cache_gpu.mortars.u_upper, cache.mortars.u_upper)
    @test_approx (cache_gpu.mortars.u_lower, cache.mortars.u_lower)

    TrixiCUDA.calc_mortar_flux!(cache_gpu.elements.surface_flux_values, mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu), equations_gpu,
                                solver_gpu.mortar, solver_gpu.surface_integral, solver_gpu, cache_gpu)
    Trixi.calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                            Trixi.have_nonconservative_terms(equations), equations,
                            solver.mortar, solver.surface_integral, solver, cache)
    @test_approx (cache_gpu.elements.surface_flux_values,
                  cache.elements.surface_flux_values)

    TrixiCUDA.calc_surface_integral!(CUDA.CUDABackend(), du_gpu, u_gpu, mesh_gpu, equations_gpu, solver_gpu.surface_integral, solver_gpu, cache_gpu)
    Trixi.calc_surface_integral!(nothing, du, u, mesh, equations, solver.surface_integral,
                                 solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.apply_jacobian!(CUDA.CUDABackend(), du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
    Trixi.apply_jacobian!(nothing, du, mesh, equations, solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.calc_sources!(CUDA.CUDABackend(), du_gpu, u_gpu, t_gpu, source_terms_gpu, equations_gpu, solver_gpu, cache_gpu)
    Trixi.calc_sources!(nothing, du, u, t, source_terms, equations, solver, cache)
    @test_approx (du_gpu, du)
end

end # module
