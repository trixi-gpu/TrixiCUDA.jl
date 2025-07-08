using Trixi, TrixiCUDA
using CUDA
using BenchmarkTools

# Set the precision
RealT = Float32

# Set up the problem
equations = ShallowWaterEquations2D(gravity_constant = 9.81f0)

function initial_condition_ec_discontinuous_bottom(x, t, element_id,
                                                   equations::ShallowWaterEquations2D)
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(RealT(0.7), RealT(0.7))
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Set the background values
    H = 4.25f0
    v1 = zero(RealT)
    v2 = zero(RealT)
    b = zero(RealT)

    # setup the discontinuous water height and velocities
    if element_id == 10
        H = 5.0f0
        v1 = RealT(0.1882) * cos_phi
        v2 = RealT(0.1882) * sin_phi
    end

    # Setup a discontinuous bottom topography using the element id number
    if element_id == 7
        b = 2 + 0.5f0 * sinpi(2 * x[1]) + 0.5f0 * cospi(2 * x[2])
    end

    return prim2cons(SVector(H, v1, v2, b), equations)
end

initial_condition = initial_condition_weak_blast_wave

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg = 4,
               surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux),
               RealT = RealT)
solver_gpu = DGSEMGPU(polydeg = 4,
                      surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
                      volume_integral = VolumeIntegralFluxDifferencing(volume_flux),
                      RealT = RealT)

coordinates_min = (-1.0f0, -1.0f0)
coordinates_max = (1.0f0, 1.0f0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 10_000,
                RealT = RealT)

# Cache initialization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver_gpu)

tspan = tspan_gpu = (0.0f0, 2.0f0)
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
# Reset initial condition on nodes
for element in eachelement(semi.solver, semi.cache)
    for j in eachnode(semi.solver), i in eachnode(semi.solver)
        x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations,
                                       semi.solver, i, j, element)
        u_node = initial_condition_ec_discontinuous_bottom(x_node, first(tspan), element,
                                                           equations)
        Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, j, element)
    end
end
du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

# ODE on GPU
ode_gpu = semidiscretizeGPU(semi_gpu, tspan_gpu)
u_gpu_ = copy(ode_gpu.u0)
du_gpu_ = similar(u_gpu_)
u_gpu = TrixiCUDA.wrap_array(u_gpu_, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
# Reset initial condition on nodes
CUDA.allowscalar(true) # disable check for scalar operations on GPU
for element in eachelement(semi_gpu.solver, semi_gpu.cache_gpu)
    for j in eachnode(semi_gpu.solver), i in eachnode(semi_gpu.solver)
        x_node = Trixi.get_node_coords(semi_gpu.cache_gpu.elements.node_coordinates, equations_gpu,
                                       semi_gpu.solver, i, j, element)
        u_node = initial_condition_ec_discontinuous_bottom(x_node, first(tspan), element,
                                                           equations)
        Trixi.set_node_vars!(u_gpu, u_node, equations_gpu, semi_gpu.solver, i, j, element)
    end
end
CUDA.allowscalar(false) # enable check for scalar operations on GPU
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
