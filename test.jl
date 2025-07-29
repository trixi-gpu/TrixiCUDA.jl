using Trixi, TrixiCUDA
using CUDA
using BenchmarkTools

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
solver_gpu = DGSEMGPU(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)
semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition_convergence_test,
                                           solver_gpu)

tspan = tspan_gpu = (0.0, 1.0)
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
@benchmark begin
    u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
    du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)
end

# ODE on GPU
# ode_gpu = semidiscretizeGPU(semi_gpu, tspan_gpu)
# u_gpu_ = copy(ode_gpu.u0)
# du_gpu_ = similar(u_gpu_)

# @benchmark begin
#     u_gpu = TrixiCUDA.wrap_array(u_gpu_, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
#     du_gpu = TrixiCUDA.wrap_array(du_gpu_, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
# end

# @benchmark begin
#     u_gpu = TrixiCUDA.wrap_array_native(u_gpu_, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
#     du_gpu = TrixiCUDA.wrap_array_native(du_gpu_, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
# end
