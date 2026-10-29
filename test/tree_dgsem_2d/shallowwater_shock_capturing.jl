module TestShallowWaterShockCapturing2D

using Trixi, TrixiShallowWater, TrixiCUDA
using Test
using CUDA

# Adapted from TrixiShallowWater.jl (66644da):
# https://github.com/trixi-framework/TrixiShallowWater.jl/blob/66644da94ebd3ad881f17379e17ffea456e15e51/examples/tree_1d_dgsem/elixir_shallowwater_shock_capturing.jl
# The 1D stone-throw setup is extended uniformly in y, with zero y velocity.
include("../test_macros.jl")

@testset "Shallow Water Shock Capturing 2D" begin
    equations = ShallowWaterEquations2D(gravity = 9.812, H0 = 1.75)

    function initial_condition_stone_throw_discontinuous_bottom(x, t,
                                                                equations::ShallowWaterEquations2D)
        H = equations.H0

        v = 0.0
        if x[1] >= -0.75 && x[1] <= 0.0
            v = -1.0
        elseif x[1] >= 0.0 && x[1] <= 0.75
            v = 1.0
        end

        b = (1.5 / exp(0.5 * ((x[1] - 1.0)^2)) +
             0.75 / exp(0.5 * ((x[1] + 1.0)^2)))

        if x[1] >= -1.5 && x[1] <= 0.0
            b = 0.5
        end

        return prim2cons(SVector(H, v, 0.0, b), equations)
    end

    initial_condition = initial_condition_stone_throw_discontinuous_bottom

    boundary_condition = boundary_condition_slip_wall

    volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
    surface_flux = (FluxHydrostaticReconstruction(FluxLaxFriedrichs(max_abs_speed_naive),
                                                  hydrostatic_reconstruction_audusse_etal),
                    flux_nonconservative_audusse_etal)
    basis = LobattoLegendreBasis(4)

    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max = 0.5,
                                             alpha_min = 0.001,
                                             alpha_smooth = true,
                                             variable = waterheight_pressure)
    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg = volume_flux,
                                                     volume_flux_fv = surface_flux)

    solver = DGSEM(basis, surface_flux, volume_integral)
    solver_gpu = DGSEMGPU(polydeg = 4, surface_flux = surface_flux,
                          volume_integral = volume_integral)

    coordinates_min = (-3.0, -3.0)
    coordinates_max = (3.0, 3.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 3,
                    periodicity = false)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions = boundary_condition)
    semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver_gpu,
                                               boundary_conditions = boundary_condition)

    tspan = tspan_gpu = (0.0, 0.1)
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
