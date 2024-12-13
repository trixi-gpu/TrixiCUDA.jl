using Trixi, TrixiCUDA
using CUDA
using BenchmarkTools

# Set up the problem
equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_weak_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_shima_etal
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver)

tspan = tspan_gpu = (0.0, 1.0)
t = t_gpu = 0.0

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

# More on custom kernels in the semidiscretization

# Get time for `rhs!` on CPU and GPU 
# Note that the first call includes compilation, and the second call will be much faster
time_cpu = @time Trixi.rhs!(du, u, t, mesh, equations, boundary_conditions, source_terms, solver, cache)
time_gpu = CUDA.@time TrixiCUDA.rhs_gpu!(du_gpu, u_gpu, t_gpu, mesh_gpu, equations_gpu,
                                         boundary_conditions_gpu, source_terms_gpu, solver_gpu, cache_gpu)

# Get benchmark for `rhs!` on CPU and GPU
bc_cpu = @benchmark Trixi.rhs!(du, u, t, mesh, equations, boundary_conditions, source_terms, solver, cache)
bc_gpu = @benchmark CUDA.@sync TrixiCUDA.rhs_gpu!(du_gpu, u_gpu, t_gpu, mesh_gpu, equations_gpu,
                                                  boundary_conditions_gpu, source_terms_gpu, solver_gpu, cache_gpu)
