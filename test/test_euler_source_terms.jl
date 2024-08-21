module TestCompressibleEulerSourceTerms

using Trixi, TrixiGPU
using OrdinaryDiffEq
using Test, CUDA

# Start testing with a clean environment
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

# Note that it is complicated to get tight error bounds for GPU kernels, so here we adopt 
# a relaxed error bound for the tests. Specifically, we use `isapprox` with the default mode, 
# i.e., `rtol = eps(Float64)^(1/2)`, to validate the precision by comparing the `Float64` 
# results from GPU kernels and CPU kernels, which corresponds to requiring equality of about 
# half of the significant digits (see https://docs.julialang.org/en/v1/base/math/#Base.isapprox).

# Basically, this heuristic method first checks whether the relaxed error bound (sometimes 
# it is further relaxed) is satisfied. Any new methods and optimizations introduced later 
# should at least satisfy this error bound.

# Test precision of the semidiscretization process
@testset "Test Compressible Euler Equation" begin
    @testset "Compressible Euler 1D" begin
        equations = CompressibleEulerEquations1D(1.4)

        initial_condition = initial_condition_convergence_test

        solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

        coordinates_min = 0.0
        coordinates_max = 2.0
        mesh = TreeMesh(coordinates_min, coordinates_max,
                        initial_refinement_level = 4,
                        n_cells_max = 10_000)

        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                            source_terms = source_terms_convergence_test)
        (; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi

        # Get copy for GPU to avoid overwriting during tests
        mesh_gpu, equations_gpu = mesh, equations
        initial_condition_gpu, boundary_conditions_gpu = initial_condition, boundary_conditions
        source_terms_gpu, solver_gpu, cache_gpu = source_terms, solver, cache

        t = 0.0
        tspan = (0.0, 2.0)

        ode = semidiscretize(semi, tspan)
        u_ode = copy(ode.u0)
        du_ode = similar(u_ode)
        u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
        du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

        # Copy data to device
        du_gpu, u_gpu = TrixiGPU.copy_to_device!(du, u)
        # Reset data on host
        Trixi.reset_du!(du, solver, cache)

        # Test `cuda_volume_integral!`
        TrixiGPU.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                       Trixi.have_nonconservative_terms(equations_gpu),
                                       equations_gpu, solver_gpu.volume_integral, solver_gpu)
        Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                    equations, solver.volume_integral, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_prolong2interfaces!`
        TrixiGPU.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
        Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_interface_flux!`
        TrixiGPU.cuda_interface_flux!(mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu),
                                      equations_gpu, solver_gpu, cache_gpu)
        Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                                   Trixi.have_nonconservative_terms(equations), equations,
                                   solver.surface_integral, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_prolong2boundaries!`
        TrixiGPU.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu, cache_gpu)
        Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_boundary_flux!`
        TrixiGPU.cuda_boundary_flux!(t, mesh_gpu, boundary_conditions_gpu, equations_gpu,
                                     solver_gpu, cache_gpu)
        Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                                  solver.surface_integral, solver)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_surface_integral!`
        TrixiGPU.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
        Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_jacobian!`
        TrixiGPU.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
        Trixi.apply_jacobian!(du, mesh, equations, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_sources!`
        TrixiGPU.cuda_sources!(du_gpu, u_gpu, t, source_terms_gpu, equations_gpu, cache_gpu)
        Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Copy data back to host
        du_cpu, u_cpu = TrixiGPU.copy_to_host!(du_gpu, u_gpu)
    end

    @testset "Compressible Euler 2D" begin
        equations = CompressibleEulerEquations2D(1.4)

        initial_condition = initial_condition_convergence_test
        solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

        coordinates_min = (0.0, 0.0)
        coordinates_max = (2.0, 2.0)
        mesh = TreeMesh(coordinates_min, coordinates_max,
                        initial_refinement_level = 4,
                        n_cells_max = 10_000)

        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                            source_terms = source_terms_convergence_test)
        (; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi

        # Get copy for GPU to avoid overwriting during tests
        mesh_gpu, equations_gpu = mesh, equations
        initial_condition_gpu, boundary_conditions_gpu = initial_condition, boundary_conditions
        source_terms_gpu, solver_gpu, cache_gpu = source_terms, solver, cache

        t = 0.0
        tspan = (0.0, 2.0)

        ode = semidiscretize(semi, tspan)
        u_ode = copy(ode.u0)
        du_ode = similar(u_ode)
        u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
        du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

        # Copy data to device
        du_gpu, u_gpu = TrixiGPU.copy_to_device!(du, u)
        # Reset data on host
        Trixi.reset_du!(du, solver, cache)

        # Test `cuda_volume_integral!`
        TrixiGPU.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                       Trixi.have_nonconservative_terms(equations_gpu),
                                       equations_gpu, solver_gpu.volume_integral, solver_gpu)
        Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                    equations, solver.volume_integral, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_prolong2interfaces!`
        TrixiGPU.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
        Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_interface_flux!`
        TrixiGPU.cuda_interface_flux!(mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu),
                                      equations_gpu, solver_gpu, cache_gpu)
        Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                                   Trixi.have_nonconservative_terms(equations), equations,
                                   solver.surface_integral, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_prolong2boundaries!`
        TrixiGPU.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu, cache_gpu)
        Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_boundary_flux!`
        TrixiGPU.cuda_boundary_flux!(t, mesh_gpu, boundary_conditions_gpu, equations_gpu,
                                     solver_gpu, cache_gpu)
        Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                                  solver.surface_integral, solver)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_surface_integral!`
        TrixiGPU.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
        Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_jacobian!`
        TrixiGPU.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
        Trixi.apply_jacobian!(du, mesh, equations, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_sources!`
        TrixiGPU.cuda_sources!(du_gpu, u_gpu, t, source_terms_gpu, equations_gpu, cache_gpu)
        Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)
        # @test_broken CUDA.@allowscalar du ≈ du_gpu 
        @test CUDA.@allowscalar isapprox(du, du_gpu, rtol = eps(Float64)^(1 / 3))
        @test CUDA.@allowscalar u ≈ u_gpu

        # Copy data back to host
        du_cpu, u_cpu = TrixiGPU.copy_to_host!(du_gpu, u_gpu)
    end

    @testset "Compressible Euler 3D" begin
        equations = CompressibleEulerEquations3D(1.4)

        initial_condition = initial_condition_convergence_test

        solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
                       volume_integral = VolumeIntegralWeakForm())

        coordinates_min = (0.0, 0.0, 0.0)
        coordinates_max = (2.0, 2.0, 2.0)
        mesh = TreeMesh(coordinates_min, coordinates_max,
                        initial_refinement_level = 2,
                        n_cells_max = 10_000)

        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                            source_terms = source_terms_convergence_test)
        (; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi

        # Get copy for GPU to avoid overwriting during tests
        mesh_gpu, equations_gpu = mesh, equations
        initial_condition_gpu, boundary_conditions_gpu = initial_condition, boundary_conditions
        source_terms_gpu, solver_gpu, cache_gpu = source_terms, solver, cache

        t = 0.0
        tspan = (0.0, 5.0)

        ode = semidiscretize(semi, tspan)
        u_ode = copy(ode.u0)
        du_ode = similar(u_ode)
        u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
        du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

        # Copy data to device
        du_gpu, u_gpu = TrixiGPU.copy_to_device!(du, u)
        # Reset data on host
        Trixi.reset_du!(du, solver, cache)

        # Test `cuda_volume_integral!`
        TrixiGPU.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                       Trixi.have_nonconservative_terms(equations_gpu),
                                       equations_gpu, solver_gpu.volume_integral, solver_gpu)
        Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                    equations, solver.volume_integral, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_prolong2interfaces!`
        TrixiGPU.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
        Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_interface_flux!`
        TrixiGPU.cuda_interface_flux!(mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu),
                                      equations_gpu, solver_gpu, cache_gpu)
        Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                                   Trixi.have_nonconservative_terms(equations), equations,
                                   solver.surface_integral, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_prolong2boundaries!`
        TrixiGPU.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu, cache_gpu)
        Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_boundary_flux!`
        TrixiGPU.cuda_boundary_flux!(t, mesh_gpu, boundary_conditions_gpu, equations_gpu,
                                     solver_gpu, cache_gpu)
        Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                                  solver.surface_integral, solver)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_surface_integral!`
        TrixiGPU.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
        Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_jacobian!`
        TrixiGPU.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
        Trixi.apply_jacobian!(du, mesh, equations, solver, cache)
        @test CUDA.@allowscalar du ≈ du_gpu
        @test CUDA.@allowscalar u ≈ u_gpu

        # Test `cuda_sources!`
        TrixiGPU.cuda_sources!(du_gpu, u_gpu, t, source_terms_gpu, equations_gpu, cache_gpu)
        Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)
        # @test_broken CUDA.@allowscalar du ≈ du_gpu 
        @test CUDA.@allowscalar isapprox(du, du_gpu, rtol = eps(Float64)^(1 / 3))
        @test CUDA.@allowscalar u ≈ u_gpu

        # Copy data back to host
        du_cpu, u_cpu = TrixiGPU.copy_to_host!(du_gpu, u_gpu)
    end
end

end # module
