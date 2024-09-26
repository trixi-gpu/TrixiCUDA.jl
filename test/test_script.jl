include("test_trixicuda.jl")

equations = ShallowWaterEquations1D(gravity_constant = 9.81)

initial_condition = initial_condition_convergence_test

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg = 3,
               surface_flux = (flux_lax_friedrichs, flux_nonconservative_fjordholm_etal),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = 0.0
coordinates_max = sqrt(2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 10_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)
semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver,
                                           source_terms = source_terms_convergence_test)

tspan = (0.0, 1.0)

# Get CPU data
(; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi

# Get GPU data
mesh_gpu, solver_gpu, cache_gpu = semi_gpu.mesh, semi_gpu.solver, semi_gpu.cache
equations_gpu, source_terms_gpu = semi_gpu.equations, semi_gpu.source_terms
initial_condition_gpu, boundary_conditions_gpu = semi_gpu.initial_condition,
                                                 semi_gpu.boundary_conditions

# Set initial time
t = t_gpu = 0.0

# Get initial data
ode = semidiscretize(semi, tspan)
u_ode = copy(ode.u0)
du_ode = similar(u_ode)
u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

# Copy data to device
du_gpu, u_gpu = TrixiCUDA.copy_to_gpu!(du, u)
# Reset data on host
Trixi.reset_du!(du, solver, cache)

# Test `cuda_volume_integral!`
TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                Trixi.have_nonconservative_terms(equations_gpu),
                                equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                cache_gpu)
Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                            equations, solver.volume_integral, solver, cache)
@test_approx du_gpu ≈ du

# Test `cuda_prolong2interfaces!`
TrixiCUDA.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
@test_approx cache_gpu.interfaces.u ≈ cache.interfaces.u

# Test `cuda_interface_flux!`
TrixiCUDA.cuda_interface_flux!(mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu),
                               equations_gpu, solver_gpu, cache_gpu)
Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                           Trixi.have_nonconservative_terms(equations), equations,
                           solver.surface_integral, solver, cache)
@test_approx cache_gpu.elements.surface_flux_values ≈ cache.elements.surface_flux_values

# Test `cuda_prolong2boundaries!`
TrixiCUDA.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu, equations_gpu,
                                   cache_gpu)
Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
@test_approx cache_gpu.boundaries.u ≈ cache.boundaries.u

# Test `cuda_boundary_flux!`
TrixiCUDA.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu,
                              Trixi.have_nonconservative_terms(equations_gpu), equations_gpu,
                              solver_gpu, cache_gpu)
Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                          solver.surface_integral, solver)
@test_approx cache_gpu.elements.surface_flux_values ≈ cache.elements.surface_flux_values

# Test `cuda_surface_integral!`
TrixiCUDA.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral, solver, cache)
@test_approx du_gpu ≈ du

# Test `cuda_jacobian!`
TrixiCUDA.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
Trixi.apply_jacobian!(du, mesh, equations, solver, cache)
@test_approx du_gpu ≈ du

# Test `cuda_sources!`
TrixiCUDA.cuda_sources!(du_gpu, u_gpu, t_gpu, source_terms_gpu, equations_gpu, cache_gpu)
Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)
@test_approx du_gpu ≈ du
