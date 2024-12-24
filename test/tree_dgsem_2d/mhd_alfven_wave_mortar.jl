module TestMHDAlfvenWaveMortar2D

using Trixi, TrixiCUDA
using Test

include("../test_macros.jl")

@testset "MHD Alfven Wave Mortar 2D" begin
    gamma = 5 / 3
    equations = IdealGlmMhdEquations2D(gamma)

    initial_condition = initial_condition_convergence_test

    volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
    solver = DGSEM(polydeg = 3,
                   surface_flux = (flux_hlle,
                                   flux_nonconservative_powell),
                   volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (0.0, 0.0)
    coordinates_max = (sqrt(2.0), sqrt(2.0))
    refinement_patches = ((type = "box", coordinates_min = 0.25 .* coordinates_max,
                           coordinates_max = 0.75 .* coordinates_max),)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 4,
                    refinement_patches = refinement_patches,
                    n_cells_max = 10_000)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
    semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver)

    tspan = tspan_gpu = (0.0, 2.0)
    t = t_gpu = 0.0

    # Semi on CPU
    (; mesh, equations, boundary_conditions, source_terms, solver, cache) = semi

    # Semi on GPU
    equations_gpu = semi_gpu.equations
    mesh_gpu, solver_gpu, cache_gpu = semi_gpu.mesh, semi_gpu.solver, semi_gpu.cache
    boundary_conditions_gpu = semi_gpu.boundary_conditions
    source_terms_gpu = semi_gpu.source_terms

    # ODE on CPU
    ode = semidiscretize(semi, tspan)
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)
    u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
    du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

    # ODE on GPU
    ode_gpu = semidiscretizeGPU(semi_gpu, tspan_gpu)
    u_gpu = copy(ode_gpu.u0)
    du_gpu = similar(u_gpu)

    # Tests for components initialization
    @test_approx (u_gpu, u)
    # du is initlaizaed as undefined, cannot test now

    # Tests for semidiscretization process
    # TrixiCUDA.reset_du!(du_gpu)
    # Trixi.reset_du!(du, solver, cache)
    # @test_approx (du_gpu, du)

    Trixi.reset_du!(du, solver, cache)

    TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                    Trixi.have_nonconservative_terms(equations_gpu),
                                    equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                    cache_gpu)
    Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                equations, solver.volume_integral, solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
    Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
    @test_approx (cache_gpu.interfaces.u, cache.interfaces.u)

    TrixiCUDA.cuda_interface_flux!(mesh_gpu,
                                   Trixi.have_nonconservative_terms(equations_gpu),
                                   equations_gpu, solver_gpu, cache_gpu)
    Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                               Trixi.have_nonconservative_terms(equations), equations,
                               solver.surface_integral, solver, cache)
    @test_approx (cache_gpu.elements.surface_flux_values,
                  cache.elements.surface_flux_values)

    TrixiCUDA.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu,
                                       equations_gpu, cache_gpu)
    Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
    @test_approx (cache_gpu.boundaries.u, cache.boundaries.u)

    TrixiCUDA.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu,
                                  Trixi.have_nonconservative_terms(equations_gpu),
                                  equations_gpu, solver_gpu, cache_gpu)
    Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                              solver.surface_integral, solver)
    @test_approx (cache_gpu.elements.surface_flux_values,
                  cache.elements.surface_flux_values)

    TrixiCUDA.cuda_prolong2mortars!(u_gpu, mesh_gpu,
                                    TrixiCUDA.check_cache_mortars(cache_gpu),
                                    solver_gpu, cache_gpu)
    Trixi.prolong2mortars!(cache, u, mesh, equations, solver.mortar,
                           solver.surface_integral, solver)
    @test_approx (cache_gpu.mortars.u_upper, cache.mortars.u_upper)
    @test_approx (cache_gpu.mortars.u_lower, cache.mortars.u_lower)

    TrixiCUDA.cuda_mortar_flux!(mesh_gpu, TrixiCUDA.check_cache_mortars(cache_gpu),
                                Trixi.have_nonconservative_terms(equations_gpu),
                                equations_gpu, solver_gpu, cache_gpu)
    Trixi.calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                            Trixi.have_nonconservative_terms(equations), equations,
                            solver.mortar, solver.surface_integral, solver, cache)
    @test_approx (cache_gpu.elements.surface_flux_values,
                  cache.elements.surface_flux_values)

    TrixiCUDA.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
    Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral,
                                 solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
    Trixi.apply_jacobian!(du, mesh, equations, solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.cuda_sources!(du_gpu, u_gpu, t_gpu, source_terms_gpu,
                            equations_gpu, cache_gpu)
    Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)
    @test_approx (du_gpu, du)
end

end # module