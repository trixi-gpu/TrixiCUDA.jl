using Trixi, TrixiGPU
using Test

# @testset "Test solver functions" begin

# The header part for testing basic kernels in 1D
advection_velocity = 1.0f0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

coordinates_min = -1.0f0
coordinates_max = 1.0f0
mesh = TreeMesh(coordinates_min,
                coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

function initial_condition_sine_wave(x, t, equations)
    SVector(1.0f0 + 0.5f0 * sinpi(sum(x - equations.advection_velocity * t)))
end

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_sine_wave, solver)

@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

t = 0.0f0
tspan = (0.0f0, 1.0f0)

ode = semidiscretize(semi, tspan)
u_ode = copy(ode.u0)
du_ode = similar(u_ode)
u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

du, u = TrixiGPU.copy_to_device!(du, u)

TrixiGPU.cuda_volume_integral!(du, u, mesh,
                               Trixi.have_nonconservative_terms(equations), equations,
                               solver.volume_integral, solver)
