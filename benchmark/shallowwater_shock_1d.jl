using Trixi, TrixiCUDA
using CUDA
using BenchmarkTools

# Set the precision
RealT = Float32

# Set up the problem
equations = ShallowWaterEquations1D(gravity_constant = 9.812f0, H0 = 1.75f0)

function initial_condition_stone_throw_discontinuous_bottom(x, t,
                                                            equations::ShallowWaterEquations1D)
    # Flat lake
    RealT = eltype(x)
    H = equations.H0

    # Discontinuous velocity
    v = zero(RealT)
    if x[1] >= -0.75f0 && x[1] <= 0
        v = -one(RealT)
    elseif x[1] >= 0 && x[1] <= 0.75f0
        v = one(RealT)
    end

    b = (1.5f0 / exp(0.5f0 * ((x[1] - 1)^2)) +
         0.75f0 / exp(0.5f0 * ((x[1] + 1)^2)))

    # Force a discontinuous bottom topography
    if x[1] >= -1.5f0 && x[1] <= 0
        b = RealT(0.5f0)
    end

    return prim2cons(SVector(H, v, b), equations)
end

initial_condition = initial_condition_stone_throw_discontinuous_bottom

boundary_condition = boundary_condition_slip_wall

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
surface_flux = (FluxHydrostaticReconstruction(flux_lax_friedrichs,
                                              hydrostatic_reconstruction_audusse_etal),
                flux_nonconservative_audusse_etal)

basis = LobattoLegendreBasis(RealT, 4)
basis_gpu = LobattoLegendreBasisGPU(4, RealT)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5f0,
                                         alpha_min = 0.001f0,
                                         alpha_smooth = true,
                                         variable = waterheight_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)
solver_gpu = DGSEMGPU(basis_gpu, surface_flux, volume_integral)

coordinates_min = -3.0f0
coordinates_max = 3.0f0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 10_000,
                periodicity = false,
                RealT = RealT)

# Cache initialization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                    solver, boundary_conditions = boundary_condition)
semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition,
                                           solver_gpu, boundary_conditions = boundary_condition)

tspan = tspan_gpu = (0.0f0, 3.0f0)
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

# Benchmark on CPU and GPU
@info "Benchmarking rhs! on CPU"
cpu_trial = @benchmark Trixi.rhs!(du, u, t, mesh, equations, boundary_conditions, source_terms,
                                  solver, cache)

@info "Benchmarking rhs! on GPU"
gpu_trial = @benchmark CUDA.@sync TrixiCUDA.rhs_gpu!(du_gpu, u_gpu, t_gpu, mesh_gpu, equations_gpu,
                                                     boundary_conditions_gpu, source_terms_gpu,
                                                     solver_gpu, cache_gpu, cache_cpu)
