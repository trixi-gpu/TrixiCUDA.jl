module TestEulerShock3D

using Trixi, TrixiCUDA
using Test
using CUDA

include("../test_macros.jl")

@testset "Euler Shock 3D" begin
    equations = CompressibleEulerEquations3D(1.4)

    initial_condition = initial_condition_weak_blast_wave

    surface_flux = flux_ranocha
    volume_flux = flux_ranocha

    polydeg = 3
    basis = LobattoLegendreBasis(polydeg)
    basis_gpu = LobattoLegendreBasisGPU(polydeg)

    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max = 0.5,
                                             alpha_min = 0.001,
                                             alpha_smooth = true,
                                             variable = density_pressure)
    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg = volume_flux,
                                                     volume_flux_fv = surface_flux)

    solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux, volume_integral = volume_integral)
    solver_gpu = DGSEMGPU(polydeg = polydeg, surface_flux = surface_flux, volume_integral = volume_integral)

    coordinates_min = (-2.0, -2.0, -2.0)
    coordinates_max = (2.0, 2.0, 2.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 3,
                    periodicity = true)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions = boundary_condition_periodic)
    semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver_gpu, boundary_conditions = boundary_condition_periodic)

    tspan = tspan_gpu = (0.0, 0.4)
    t = t_gpu = 0.0

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
    @test_approx (u_gpu, u) # du is initlaizaed as undefined, cannot test now
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
    @test_approx (cache_gpu.mortars.u_upper_left, cache.mortars.u_upper_left)
    @test_approx (cache_gpu.mortars.u_upper_right, cache.mortars.u_upper_right)
    @test_approx (cache_gpu.mortars.u_lower_left, cache.mortars.u_lower_left)
    @test_approx (cache_gpu.mortars.u_lower_right, cache.mortars.u_lower_right)

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
