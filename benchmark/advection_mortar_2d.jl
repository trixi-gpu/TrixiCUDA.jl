using Trixi, TrixiCUDA
using CUDA
using BenchmarkTools

# Set the precision
RealT = Float32

# Set up the problem
advection_velocity = (0.2f0, -0.7f0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs, RealT = RealT)
solver_gpu = DGSEMGPU(polydeg = 3, surface_flux = flux_lax_friedrichs, RealT = RealT)

coordinates_min = (-1.0f0, -1.0f0)
coordinates_max = (1.0f0, 1.0f0)
refinement_patches = ((type = "box", coordinates_min = (0.0f0, -1.0f0),
                       coordinates_max = (1.0f0, 1.0f0)),)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                refinement_patches = refinement_patches,
                n_cells_max = 10_000, RealT = RealT)

# Cache initialization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver_gpu)

tspan = tspan_gpu = (0.0f0, 1.0f0)
t = t_gpu = 0.0f0

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

# Warm up
Trixi.rhs!(du, u, t, mesh, equations, boundary_conditions, source_terms, solver, cache)
TrixiCUDA.rhs_gpu!(du_gpu, u_gpu, t_gpu, mesh_gpu, equations_gpu, boundary_conditions_gpu,
                   source_terms_gpu, solver_gpu, cache_gpu, cache_cpu)

# Get DOFs (per field)
dofs = Trixi.ndofsglobal(semi)

# Benchmark on CPU and GPU
@info "Benchmarking rhs! on CPU"
cpu_trial = @benchmark Trixi.rhs!(du, u, t, mesh, equations, boundary_conditions, source_terms,
                                  solver, cache)

@info "Benchmarking rhs! on GPU"
gpu_trial = @benchmark CUDA.@sync TrixiCUDA.rhs_gpu!(du_gpu, u_gpu, t_gpu, mesh_gpu, equations_gpu,
                                                     boundary_conditions_gpu, source_terms_gpu,
                                                     solver_gpu, cache_gpu, cache_cpu)
