using Trixi, TrixiGPU
using OrdinaryDiffEq
using CUDA
using Test

equations = ShallowWaterEquations1D(gravity_constant = 9.81, H0 = 3.25)

function initial_condition_discontinuous_well_balancedness(x, t,
                                                           equations::ShallowWaterEquations1D)
    H = equations.H0
    v = 0.0
    b = 0.0

    if x[1] >= 0.5 && x[1] <= 0.75
        b = 2.0 + 0.5 * sin(2.0 * pi * x[1])
    end

    return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_discontinuous_well_balancedness

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal)
solver = DGSEM(polydeg = 4, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = -1.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level = 3,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
(; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi

# Get copy for GPU to avoid overwriting during tests
mesh_gpu, equations_gpu = mesh, equations
initial_condition_gpu, boundary_conditions_gpu = initial_condition, boundary_conditions
source_terms_gpu, solver_gpu, cache_gpu = source_terms, solver, cache

t = t_gpu = 0.0
tspan = (0.0, 100.0)

ode = semidiscretize(semi, tspan)
u_ode = copy(ode.u0)
du_ode = similar(u_ode)
u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

# Copy data to device
du_gpu, u_gpu = TrixiGPU.copy_to_device!(du, u)
# Reset data on host
Trixi.reset_du!(du, solver, cache)

# Test `cuda_volume_integral!`
TrixiGPU.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                               Trixi.have_nonconservative_terms(equations_gpu),
                               equations_gpu, solver_gpu.volume_integral, solver_gpu)
Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                            equations, solver.volume_integral, solver, cache)
@test CUDA.@allowscalar du ≈ du_gpu
@test CUDA.@allowscalar u ≈ u_gpu

# Test `cuda_prolong2interfaces!`
TrixiGPU.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
@test CUDA.@allowscalar du ≈ du_gpu
@test CUDA.@allowscalar u ≈ u_gpu

# Test `cuda_interface_flux!`
TrixiGPU.cuda_interface_flux!(mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu),
                              equations_gpu, solver_gpu, cache_gpu)
Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                           Trixi.have_nonconservative_terms(equations), equations,
                           solver.surface_integral, solver, cache)
@test CUDA.@allowscalar du ≈ du_gpu
@test CUDA.@allowscalar u ≈ u_gpu

# Test `cuda_prolong2boundaries!`
TrixiGPU.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu, equations_gpu,
                                  cache_gpu)
Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
@test CUDA.@allowscalar du ≈ du_gpu
@test CUDA.@allowscalar u ≈ u_gpu

# Test `cuda_boundary_flux!`
TrixiGPU.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu, equations_gpu,
                             solver_gpu, cache_gpu)
Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                          solver.surface_integral, solver)
@test CUDA.@allowscalar du ≈ du_gpu
@test CUDA.@allowscalar u ≈ u_gpu

# Test `cuda_surface_integral!`
TrixiGPU.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral, solver, cache)
@test CUDA.@allowscalar du ≈ du_gpu
@test CUDA.@allowscalar u ≈ u_gpu

# Test `cuda_jacobian!`
TrixiGPU.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
Trixi.apply_jacobian!(du, mesh, equations, solver, cache)
@test CUDA.@allowscalar du ≈ du_gpu
@test CUDA.@allowscalar u ≈ u_gpu

# Test `cuda_sources!`
TrixiGPU.cuda_sources!(du_gpu, u_gpu, t_gpu, source_terms_gpu, equations_gpu, cache_gpu)
Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)
@test CUDA.@allowscalar du ≈ du_gpu
@test CUDA.@allowscalar u ≈ u_gpu
