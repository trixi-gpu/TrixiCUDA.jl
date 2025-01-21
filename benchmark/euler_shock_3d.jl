using Trixi, TrixiCUDA
using CUDA
using BenchmarkTools

# Set the precision
RealT = Float32

# Set up the problem
equations = CompressibleEulerEquations3D(1.4f0)

initial_condition = initial_condition_weak_blast_wave

surface_flux = flux_ranocha
volume_flux = flux_ranocha

polydeg = 3
basis = LobattoLegendreBasis(RealT, polydeg)
basis_gpu = LobattoLegendreBasisGPU(polydeg, RealT)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5f0,
                                         alpha_min = 0.001f0,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)
solver_gpu = DGSEMGPU(basis_gpu, surface_flux, volume_integral)

coordinates_min = (-2.0f0, -2.0f0, -2.0f0)
coordinates_max = (2.0f0, 2.0f0, 2.0f0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 100_000, RealT = RealT)

# Cache initialization
@info "Time for cache initialization on CPU"
@time semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                          source_terms = source_terms_convergence_test)
@info "Time for cache initialization on GPU"
CUDA.@time semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver_gpu,
                                                      source_terms = source_terms_convergence_test)

tspan = tspan_gpu = (0.0f0, 0.4f0)
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
u_gpu_ = copy(ode_gpu.u0)
du_gpu_ = similar(u_gpu_)
u_gpu = TrixiCUDA.wrap_array(u_gpu_, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
du_gpu = TrixiCUDA.wrap_array(du_gpu_, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)

# Reset du and volume integral
@info "Time for reset_du! and volume_integral! on GPU"
@benchmark CUDA.@sync TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                                      Trixi.have_nonconservative_terms(equations_gpu),
                                                      equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                                      cache_gpu)
