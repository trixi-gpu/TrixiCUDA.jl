# Include the test macros
include("test_macros.jl")

# Test cuda_copy_to_gpu!
function test_copy_to_gpu(du_gpu, u_gpu, du, u, solver, cache)
    du_gpu, u_gpu = TrixiCUDA.copy_to_gpu!(du, u)
    Trixi.reset_du!(du, solver, cache)
    @test_approx du_gpu ≈ du
    # @test_equal u_gpu ≈ u
end

# Test cuda_volume_integral!
function test_volume_integral(du_gpu, u_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu,
                              du, u, mesh, equations, solver, cache)
    TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                    Trixi.have_nonconservative_terms(equations_gpu), equations_gpu,
                                    solver_gpu.volume_integral, solver_gpu, cache_gpu)
    Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations), equations,
                                solver.volume_integral, solver, cache)
    @test_approx du_gpu ≈ du
end

# Test cuda_prolong2interfaces!
function test_prolong2interfaces(u_gpu, mesh_gpu, equations_gpu, cache_gpu,
                                 u, mesh, equations, solver, cache)
    TrixiCUDA.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
    Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
    @test_approx cache_gpu.interfaces.u ≈ cache.interfaces.u
end

# Test cuda_interface_flux!
function test_interface_flux(mesh_gpu, equations_gpu, solver_gpu, cache_gpu,
                             mesh, equations, solver, cache)
    TrixiCUDA.cuda_interface_flux!(mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu),
                                   equations_gpu, solver_gpu, cache_gpu)
    Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                               Trixi.have_nonconservative_terms(equations), equations,
                               solver.surface_integral, solver, cache)
    @test_approx cache_gpu.elements.surface_flux_values ≈ cache.elements.surface_flux_values
end

# Test cuda_prolong2boundaries!
function test_prolong2boundaries(u_gpu, mesh_gpu, boundary_conditions_gpu, equations_gpu, cache_gpu,
                                 u, mesh, equations, solver, cache)
    TrixiCUDA.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu, equations_gpu,
                                       cache_gpu)
    Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
    @test_approx cache_gpu.boundaries.u ≈ cache.boundaries.u
end

# Test cuda_boundary_flux!
function test_boundary_flux(t_gpu, mesh_gpu, boundary_conditions_gpu, equations_gpu, solver_gpu,
                            cache_gpu, t, mesh, equations, solver, cache)
    TrixiCUDA.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu,
                                  Trixi.have_nonconservative_terms(equations_gpu), equations_gpu,
                                  solver_gpu, cache_gpu)
    Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                              solver.surface_integral, solver)
    @test_approx cache_gpu.elements.surface_flux_values ≈ cache.elements.surface_flux_values
end

# Test cuda_surface_integral!
function test_surface_integral(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu,
                               du, u, mesh, equations, solver, cache)
    TrixiCUDA.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
    Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral, solver, cache)
    @test_approx du_gpu ≈ du
end

# Test cuda_jacobian!
function test_jacobian(du_gpu, mesh_gpu, equations_gpu, cache_gpu,
                       du, mesh, equations, solver, cache)
    TrixiCUDA.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
    Trixi.apply_jacobian!(du, mesh, equations, solver, cache)
    @test_approx du_gpu ≈ du
end

# Test cuda_sources!
function test_sources(du_gpu, u_gpu, t_gpu, source_terms_gpu, equations_gpu, cache_gpu,
                      du, u, t, source_terms, equations, solver, cache)
    TrixiCUDA.cuda_sources!(du_gpu, u_gpu, t_gpu, source_terms_gpu, equations_gpu, cache_gpu)
    Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)
    @test_approx du_gpu ≈ du
end

# Test cuda_copy_to_cpu!
function test_copy_to_cpu(du_gpu, u_gpu, du, u)
    du_cpu, u_cpu = TrixiCUDA.copy_to_cpu!(du_gpu, u_gpu)
    @test_approx du_cpu ≈ du
    @test_approx u_cpu ≈ u
end
