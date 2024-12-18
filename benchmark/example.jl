using Trixi, TrixiCUDA
using CUDA
using BenchmarkTools

# Set the precision
RealT = Float32

equations = IdealGlmMhdEquations3D(RealT(5 / 3))

initial_condition = initial_condition_convergence_test

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3,
               surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux), RealT = RealT)

coordinates_min = (-1.0f0, -1.0f0, -1.0f0)
coordinates_max = (1.0f0, 1.0f0, 1.0f0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 10_000, RealT = RealT)

# Cache initialization
@info "Time for cache initialization on CPU"
@time semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
@info "Time for cache initialization on GPU"
CUDA.@time semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver)

tspan = tspan_gpu = (0.0f0, 1.0f0)
t = t_gpu = 0.0f0

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

# Reset du
@info "Time for reset_du! on GPU"
CUDA.@time TrixiCUDA.reset_du!(du_gpu)
@info "Time for reset_du! on CPU"
@time Trixi.reset_du!(du, solver, cache)

# Volume integral
@info "Time for volume_integral! on GPU"
CUDA.@time TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                           Trixi.have_nonconservative_terms(equations_gpu),
                                           equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                           cache_gpu)
@info "Time for volume_integral! on CPU"
@time Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                  equations, solver.volume_integral, solver, cache)

# Prolong to interfaces
@info "Time for prolong2interfaces! on GPU"
CUDA.@time TrixiCUDA.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
@info "Time for prolong2interfaces! on CPU"
@time Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)

# Interface flux
@info "Time for interface_flux! on GPU"
CUDA.@time TrixiCUDA.cuda_interface_flux!(mesh_gpu,
                                          Trixi.have_nonconservative_terms(equations_gpu),
                                          equations_gpu, solver_gpu, cache_gpu)
@info "Time for interface_flux! on CPU"
@time Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                                 Trixi.have_nonconservative_terms(equations), equations,
                                 solver.surface_integral, solver, cache)

# Prolong to boundaries
@info "Time for prolong2boundaries! on GPU"
CUDA.@time TrixiCUDA.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu,
                                              equations_gpu, cache_gpu)
@info "Time for prolong2boundaries! on CPU"
@time Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)

# Boundary flux
@info "Time for boundary_flux! on GPU"
CUDA.@time TrixiCUDA.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu,
                                         Trixi.have_nonconservative_terms(equations_gpu),
                                         equations_gpu, solver_gpu, cache_gpu)
@info "Time for boundary_flux! on CPU"
@time Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                                solver.surface_integral, solver)

# Prolong to mortars
@info "Time for prolong2mortars! on GPU"
CUDA.@time TrixiCUDA.cuda_prolong2mortars!(u_gpu, mesh_gpu,
                                           TrixiCUDA.check_cache_mortars(cache_gpu),
                                           solver_gpu, cache_gpu)
@info "Time for prolong2mortars! on CPU"
@time Trixi.prolong2mortars!(cache, u, mesh, equations,
                             solver.mortar, solver.surface_integral, solver)

# Mortar flux
@info "Time for mortar_flux! on GPU"
CUDA.@time TrixiCUDA.cuda_mortar_flux!(mesh_gpu, TrixiCUDA.check_cache_mortars(cache_gpu),
                                       Trixi.have_nonconservative_terms(equations_gpu),
                                       equations_gpu, solver_gpu, cache_gpu)
@info "Time for mortar_flux! on CPU"
@time Trixi.calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                              Trixi.have_nonconservative_terms(equations), equations,
                              solver.mortar, solver.surface_integral, solver, cache)

# Surface integral
@info "Time for surface_integral! on GPU"
CUDA.@time TrixiCUDA.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
@info "Time for surface_integral! on CPU"
@time Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral,
                                   solver, cache)

# Jacobian
@info "Time for jacobian! on GPU"
CUDA.@time TrixiCUDA.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
@info "Time for jacobian! on CPU"
@time Trixi.apply_jacobian!(du, mesh, equations, solver, cache)

# Sources terms
@info "Time for sources! on GPU"
CUDA.@time TrixiCUDA.cuda_sources!(du_gpu, u_gpu, t_gpu, source_terms_gpu,
                                   equations_gpu, cache_gpu)
@info "Time for sources! on CPU"
@time Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)

# Semidiscretization process
@info "Time for rhs! on GPU"
CUDA.@time TrixiCUDA.rhs_gpu!(du_gpu, u_gpu, t_gpu, mesh_gpu, equations_gpu,
                              boundary_conditions_gpu, source_terms_gpu,
                              solver_gpu, cache_gpu)
@info "Time for rhs! on CPU"
@time Trixi.rhs!(du, u, t, mesh, equations, boundary_conditions, source_terms,
                 solver, cache)
