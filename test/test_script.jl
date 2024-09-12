using Trixi, TrixiGPU
using OrdinaryDiffEq
using CUDA
using Test

equations = ShallowWaterEquations1D(gravity_constant = 9.812, H0 = 1.75)

# Initial condition with a truly discontinuous velocity and bottom topography.
# Works as intended for TreeMesh1D with `initial_refinement_level=3`. If the mesh
# refinement level is changed the initial condition below may need changed as well to
# ensure that the discontinuities lie on an element interface.
function initial_condition_stone_throw_discontinuous_bottom(x, t,
                                                            equations::ShallowWaterEquations1D)

    # Calculate primitive variables

    # flat lake
    H = equations.H0

    # Discontinuous velocity
    v = 0.0
    if x[1] >= -0.75 && x[1] <= 0.0
        v = -1.0
    elseif x[1] >= 0.0 && x[1] <= 0.75
        v = 1.0
    end

    b = (1.5 / exp(0.5 * ((x[1] - 1.0)^2)) +
         0.75 / exp(0.5 * ((x[1] + 1.0)^2)))

    # Force a discontinuous bottom topography
    if x[1] >= -1.5 && x[1] <= 0.0
        b = 0.5
    end

    return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_stone_throw_discontinuous_bottom

boundary_condition = boundary_condition_slip_wall

###############################################################################
# Get the DG approximation space

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_lax_friedrichs,
                                              hydrostatic_reconstruction_audusse_etal),
                flux_nonconservative_audusse_etal)
basis = LobattoLegendreBasis(4)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = waterheight_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Create the TreeMesh for the domain [-3, 3]

coordinates_min = -3.0
coordinates_max = 3.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 10_000,
                periodicity = false)

# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solver

tspan = (0.0, 3.0)

# Get CPU data
(; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi

# Get GPU data
equations_gpu = deepcopy(equations)
mesh_gpu, solver_gpu, cache_gpu = deepcopy(mesh), deepcopy(solver), deepcopy(cache)
boundary_conditions_gpu, source_terms_gpu = deepcopy(boundary_conditions),
                                            deepcopy(source_terms)

# Set initial time
t = t_gpu = 0.0

# Get initial data
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
                               equations_gpu, solver_gpu.volume_integral, solver_gpu,
                               cache_gpu)
Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                            equations, solver.volume_integral, solver, cache)
@test CUDA.@allowscalar du_gpu ≈ du
