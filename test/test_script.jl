include("test_macros.jl")

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

ode = semidiscretize(semi, tspan)
u_ode = copy(ode.u0)
du_ode = similar(u_ode)

# Get CPU data
t = 0.0
(; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi
u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

# Get GPU data
t_gpu = 0.0
equations_gpu = semi_gpu.equations
mesh_gpu, solver_gpu, cache_gpu = semi_gpu.mesh, semi_gpu.solver, semi_gpu.cache
initial_condition_gpu = semi_gpu.initial_condition
boundary_conditions_gpu = semi_gpu.boundary_conditions
source_terms_gpu = semi_gpu.source_terms
u_gpu = CuArray(u)
du_gpu = CuArray(du)

# Copy data to device
du_gpu, u_gpu = TrixiCUDA.copy_to_gpu!(du, u)
# Reset data on host
Trixi.reset_du!(du, solver, cache)

derivative_dhat = CuArray{Float64}(solver_gpu.basis.derivative_dhat)
flux_arr1 = similar(u_gpu)
flux_arr2 = similar(u_gpu)
flux_arr3 = similar(u_gpu)

size_arr = CuArray{Float64}(undef, size(u, 2)^3, size(u, 5))

flux_kernel = @cuda launch=false TrixiCUDA.flux_kernel!(flux_arr1, flux_arr2, flux_arr3, u_gpu,
                                                        equations,
                                                        flux)

config = launch_configuration(flux_kernel.fun)
