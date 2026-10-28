module TestShallowWaterEc2D

using Trixi, TrixiCUDA, TrixiShallowWater
using Test

# Adapted from TrixiShallowWater.jl (66644da):
# https://github.com/trixi-framework/TrixiShallowWater.jl/blob/66644da94ebd3ad881f17379e17ffea456e15e51/examples/tree_2d_dgsem/elixir_shallowwater_ec.jl
include("../test_macros.jl")

@testset "Shallow Water EC 2D" begin
    equations = ShallowWaterEquations2D(gravity = 9.81)

    initial_condition = initial_condition_weak_blast_wave

    volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
    solver = DGSEM(polydeg = 4,
                   surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
                   volume_integral = VolumeIntegralFluxDifferencing(volume_flux))
    solver_gpu = DGSEMGPU(polydeg = 4,
                          surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
                          volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-1.0, -1.0)
    coordinates_max = (1.0, 1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 2,
                    periodicity = true)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions = boundary_condition_periodic)
    semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver_gpu,
                                               boundary_conditions = boundary_condition_periodic)

    function initial_condition_ec_discontinuous_bottom(x, t, element_id,
                                                       equations::ShallowWaterEquations2D)
        inicenter = SVector(0.7, 0.7)
        x_norm = x[1] - inicenter[1]
        y_norm = x[2] - inicenter[2]
        r = sqrt(x_norm^2 + y_norm^2)
        phi = atan(y_norm, x_norm)
        sin_phi, cos_phi = sincos(phi)

        H = 4.25
        v1 = 0.0
        v2 = 0.0
        b = 0.0

        if element_id == 10
            H = 5.0
            v1 = 0.1882 * cos_phi
            v2 = 0.1882 * sin_phi
        end

        if element_id == 7
            b = 2.0 + 0.5 * sin(2.0 * pi * x[1]) + 0.5 * cos(2.0 * pi * x[2])
        end

        return prim2cons(SVector(H, v1, v2, b), equations)
    end

    function initialize_ec!(u, semi)
        equations = semi.equations
        for element in eachelement(semi.solver, semi.cache)
            for j in eachnode(semi.solver), i in eachnode(semi.solver)
                x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations,
                                               semi.solver, i, j, element)
                u_node = initial_condition_ec_discontinuous_bottom(x_node, 0.0, element,
                                                                   equations)
                Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, j, element)
            end
        end
        @test any(!iszero, view(u, 4, :, :, 7))
        @test all(iszero, view(u, 4, :, :, 10))
        @test u[1, 1, 1, 10] ≈ 5.0
        return nothing
    end

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

    # Apply the element-dependent initial condition to both solution buffers.
    initialize_ec!(u, semi)
    u_host = Array(u_gpu)
    initialize_ec!(u_host, semi)
    copyto!(u_gpu, u_host)

    # Tests for semidiscretization process
    @test_approx (u_gpu, u)
    @test all(isfinite, Array(u_gpu))
    Trixi.set_zero!(du, solver, cache)
    # Boundary faces have not been filled when comparing the interface fluxes.
    fill!(cache.elements.surface_flux_values, 0.0)
    fill!(cache_gpu.elements.surface_flux_values, 0.0)

    TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                    Trixi.have_nonconservative_terms(equations_gpu),
                                    equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                    cache_gpu, cache_cpu)
    Trixi.calc_volume_integral!(nothing, du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                equations, solver.volume_integral, solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
    Trixi.prolong2interfaces!(nothing, cache, u, mesh, equations, solver)
    @test_approx (cache_gpu.interfaces.u, cache.interfaces.u)

    TrixiCUDA.cuda_interface_flux!(mesh_gpu,
                                   Trixi.have_nonconservative_terms(equations_gpu),
                                   equations_gpu, solver_gpu, cache_gpu)
    Trixi.calc_interface_flux!(nothing, cache.elements.surface_flux_values, mesh,
                               Trixi.have_nonconservative_terms(equations), equations,
                               solver.surface_integral, solver, cache)
    @test_approx (cache_gpu.elements.surface_flux_values,
                  cache.elements.surface_flux_values)

    TrixiCUDA.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu,
                                       equations_gpu, cache_gpu)
    Trixi.prolong2boundaries!(nothing, cache, u, mesh, equations, solver)
    @test_approx (cache_gpu.boundaries.u, cache.boundaries.u)

    TrixiCUDA.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu,
                                  Trixi.have_nonconservative_terms(equations_gpu),
                                  equations_gpu, solver_gpu, cache_gpu)
    Trixi.calc_boundary_flux!(nothing, cache, t, boundary_conditions, mesh, equations,
                              solver.surface_integral, solver)
    @test_approx (cache_gpu.elements.surface_flux_values,
                  cache.elements.surface_flux_values)

    TrixiCUDA.cuda_prolong2mortars!(u_gpu, mesh_gpu,
                                    TrixiCUDA.check_cache_mortars(cache_gpu),
                                    solver_gpu, cache_gpu)
    Trixi.prolong2mortars!(cache, u, mesh, equations, solver.mortar, solver)
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
    Trixi.calc_surface_integral!(nothing, du, u, mesh, equations, solver.surface_integral,
                                 solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
    Trixi.apply_jacobian!(nothing, du, mesh, equations, solver, cache)
    @test_approx (du_gpu, du)

    TrixiCUDA.cuda_sources!(du_gpu, u_gpu, t_gpu, source_terms_gpu,
                            equations_gpu, cache_gpu)
    Trixi.calc_sources!(nothing, du, u, t, source_terms, equations, solver, cache)
    @test_approx (du_gpu, du)
    @test all(isfinite, Array(du_gpu))
    @test all(iszero, selectdim(Array(du_gpu), 1, 4))
end

end # module
