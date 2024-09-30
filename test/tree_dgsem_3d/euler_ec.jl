module TestEulerEC3D

include("../test_macros.jl")

@testset "Euler EC 3D" begin
    equations = CompressibleEulerEquations3D(1.4)

    initial_condition = initial_condition_weak_blast_wave

    volume_flux = flux_ranocha
    solver = DGSEM(polydeg = 3, surface_flux = flux_ranocha,
                   volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-2.0, -2.0, -2.0)
    coordinates_max = (2.0, 2.0, 2.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 3,
                    n_cells_max = 100_000)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
    semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver)

    tspan = (0.0, 0.4)

    ode = semidiscretize(semi, tspan)
    u_ode = copy(ode.u0)
    du_ode = similar(u_ode)

    # Get CPU data
    t = 0.0
    (; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi
    u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
    du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

    # Get GPU data
    t_gpu = 0.0
    equations_gpu = semi_gpu.equations
    mesh_gpu, solver_gpu, cache_gpu = semi_gpu.mesh, semi_gpu.solver, semi_gpu.cache
    initial_condition_gpu = semi_gpu.initial_condition
    boundary_conditions_gpu = semi_gpu.boundary_conditions
    source_terms_gpu = semi_gpu.source_terms
    u_gpu = CuArray(u)
    du_gpu = CuArray(du)

    # Begin tests
    @testset "Semidiscretization Process" begin
        @testset "Copy to GPU" begin
            du_gpu, u_gpu = TrixiCUDA.copy_to_gpu!(du, u)
            Trixi.reset_du!(du, solver, cache)
            @test_approx (du_gpu, du)
            @test_equal (u_gpu, u)
        end

        @testset "Volume Integral" begin
            TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                            Trixi.have_nonconservative_terms(equations_gpu),
                                            equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                            cache_gpu)
            Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                        equations, solver.volume_integral, solver, cache)
            @test_approx (du_gpu, du)
            @test_equal (u_gpu, u)
        end

        @testset "Prolong Interfaces" begin
            TrixiCUDA.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
            Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
            @test_approx (cache_gpu.interfaces.u, cache.interfaces.u)
            @test_equal (u_gpu, u)
        end

        @testset "Interface Flux" begin
            TrixiCUDA.cuda_interface_flux!(mesh_gpu,
                                           Trixi.have_nonconservative_terms(equations_gpu),
                                           equations_gpu, solver_gpu, cache_gpu)
            Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                                       Trixi.have_nonconservative_terms(equations), equations,
                                       solver.surface_integral, solver, cache)
            @test_approx (cache_gpu.elements.surface_flux_values,
                          cache.elements.surface_flux_values)
            @test_equal (u_gpu, u)
        end

        @testset "Prolong Boundaries" begin
            TrixiCUDA.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu,
                                               equations_gpu, cache_gpu)
            Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
            @test_approx (cache_gpu.boundaries.u, cache.boundaries.u)
            @test_equal (u_gpu, u)
        end

        @testset "Boundary Flux" begin
            TrixiCUDA.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu,
                                          Trixi.have_nonconservative_terms(equations_gpu),
                                          equations_gpu, solver_gpu, cache_gpu)
            Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                                      solver.surface_integral, solver)
            @test_approx (cache_gpu.elements.surface_flux_values,
                          cache.elements.surface_flux_values)
            @test_equal (u_gpu, u)
        end

        @testset "Prolong mortars" begin
            TrixiCUDA.cuda_prolong2mortars!(u_gpu, mesh_gpu,
                                            TrixiCUDA.check_cache_mortars(cache_gpu),
                                            solver_gpu, cache_gpu)
            Trixi.prolong2mortars!(cache, u, mesh, equations,
                                   solver.mortar, solver.surface_integral, solver)
            @test_approx (cache_gpu.mortars.u_upper_left, cache.mortars.u_upper_left)
            @test_approx (cache_gpu.mortars.u_upper_right, cache.mortars.u_upper_right)
            @test_approx (cache_gpu.mortars.u_lower_left, cache.mortars.u_lower_left)
            @test_approx (cache_gpu.mortars.u_lower_right, cache.mortars.u_lower_right)
            @test_equal (u_gpu, u)
        end

        @testset "Mortar Flux" begin
            TrixiCUDA.cuda_mortar_flux!(mesh_gpu, TrixiCUDA.check_cache_mortars(cache_gpu),
                                        Trixi.have_nonconservative_terms(equations_gpu),
                                        equations_gpu, solver_gpu, cache_gpu)
            Trixi.calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                                    Trixi.have_nonconservative_terms(equations), equations,
                                    solver.mortar, solver.surface_integral, solver, cache)
            @test_approx (cache_gpu.elements.surface_flux_values,
                          cache.elements.surface_flux_values)
            @test_equal (u_gpu, u)
        end

        @testset "Surface Integral" begin
            TrixiCUDA.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
            Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral,
                                         solver, cache)
            @test_approx (du_gpu, du)
            @test_equal (u_gpu, u)
        end

        @testset "Apply Jacobian" begin
            TrixiCUDA.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
            Trixi.apply_jacobian!(du, mesh, equations, solver, cache)
            @test_approx (du_gpu, du)
            @test_equal (u_gpu, u)
        end

        @testset "Apply Sources" begin
            TrixiCUDA.cuda_sources!(du_gpu, u_gpu, t_gpu, source_terms_gpu,
                                    equations_gpu, cache_gpu)
            Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)
            @test_approx (du_gpu, du)
            @test_equal (u_gpu, u)
        end

        @testset "Copy to CPU" begin
            du_cpu, u_cpu = TrixiCUDA.copy_to_cpu!(du_gpu, u_gpu)
            @test_approx (du_cpu, du)
            @test_equal (u_cpu, u)
        end
    end
end

end # module
