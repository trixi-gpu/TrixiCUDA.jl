using Trixi, TrixiCUDA
using CUDA
using BenchmarkTools

# Set the precision
RealT = Float32

# Set up the problem
equations = IdealGlmMhdEquations3D(1.4f0)

initial_condition = initial_condition_weak_blast_wave

volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3,
               surface_flux = (flux_hindenlang_gassner, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux),
               RealT = RealT)
solver_gpu = DGSEMGPU(polydeg = 3,
                      surface_flux = (flux_hindenlang_gassner, flux_nonconservative_powell),
                      volume_integral = VolumeIntegralFluxDifferencing(volume_flux),
                      RealT = RealT)

coordinates_min = (-2.0f0, -2.0f0, -2.0f0)
coordinates_max = (2.0f0, 2.0f0, 2.0f0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 10_000, RealT = RealT)

# Cache initialization
@info "Time for cache initialization on CPU"
@time semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
@info "Time for cache initialization on GPU"
CUDA.@time semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver_gpu)

tspan_gpu = (0.0f0, 0.4f0)
t_gpu = 0.0f0

# Semi on GPU
equations_gpu, mesh_gpu, solver_gpu = semi_gpu.equations, semi_gpu.mesh, semi_gpu.solver
cache_gpu, cache_cpu = semi_gpu.cache_gpu, semi_gpu.cache_cpu
boundary_conditions_gpu, source_terms_gpu = semi_gpu.boundary_conditions, semi_gpu.source_terms

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
                                                      cache_gpu, cache_cpu)
