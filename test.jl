using Trixi, TrixiCUDA
using Test
include("test/test_macros.jl")

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_weak_blast_wave

volume_flux = flux_ranocha
solver = DGSEM(polydeg = 5, surface_flux = flux_ranocha,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-2.0, -2.0, -2.0)
coordinates_max = (2.0, 2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver)

tspan = tspan_gpu = (0.0, 0.4)
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

# Tests for components initialization
@test_approx (u_gpu, u)
# du is initlaizaed as undefined, cannot test now

Trixi.reset_du!(du, solver, cache)

TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                Trixi.have_nonconservative_terms(equations_gpu),
                                equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                cache_gpu)
Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                            equations, solver.volume_integral, solver, cache)
@test_approx (du_gpu, du)
