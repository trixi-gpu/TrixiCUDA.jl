include("test_macros.jl")

using Trixi, TrixiCUDA
using Test

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)
semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition_convergence_test,
                                           solver)

tspan = (0.0, 1.0)
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
ode_gpu = semidiscretizeGPU(semi_gpu, tspan)
u_gpu = copy(ode_gpu.u0)
du_gpu = similar(u_gpu)

# Tests for components initialization
@test_approx (u_gpu, u)

# Test `rhs!` function as a whole 
Trixi.rhs!(du, u, t, mesh, equations, boundary_conditions, source_terms, solver, cache)
TrixiCUDA.rhs_gpu!(du_gpu, u_gpu, t_gpu, mesh_gpu, equations_gpu, boundary_conditions_gpu,
                   source_terms_gpu, solver_gpu, cache_gpu)
@test_approx (du_gpu, du)
