#include("header.jl") # Remove it after first run to avoid recompilation

# Set random seed
Random.seed!(12345)

# The header part of test
equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test

# Note that the expected EOC of 5 is not reached with this flux.
# Using flux_hll instead yields the expected EOC.
solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs)

coordinates_min = 0.0
coordinates_max = 2.0
mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=4,
    n_cells_max=10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
    source_terms=source_terms_convergence_test)

@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

l = nvariables(equations) * nnodes(solver)^ndims(mesh) * nelements(solver, cache)
u_ode = rand(Float64, l)
du_ode = rand(Float64, l)
u = wrap_array(u_ode, mesh, equations, solver, cache)
du = wrap_array(du_ode, mesh, equations, solver, cache)

# CUDA kernel configurators for 1D, 2D, and 3D arrays
#################################################################################

# CUDA kernel configurator for 1D array computing
function configurator_1d(kernel::CUDA.HostKernel, array::CuArray{Float32})
    config = launch_configuration(kernel.fun)

    threads = min(length(array), config.threads)
    blocks = cld(length(array), threads)

    return (threads=threads, blocks=blocks)
end

# CUDA kernel configurator for 2D array computing
function configurator_2d(kernel::CUDA.HostKernel, array::CuArray{Float32})
    config = launch_configuration(kernel.fun)

    threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 2))), 2))
    blocks = map(cld, size(array), threads)

    return (threads=threads, blocks=blocks)
end

# CUDA kernel configurator for 3D array computing
function configurator_3d(kernel::CUDA.HostKernel, array::CuArray{Float32})
    config = launch_configuration(kernel.fun)

    threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 3))), 3))
    blocks = map(cld, size(array), threads)

    return (threads=threads, blocks=blocks)
end

# Rewrite `rhs!()` from `trixi/src/solvers/dgsem_tree/dg_1d.jl`
#################################################################################

# Copy `du` and `u` to GPU (run as Float32)
function copy_to_gpu!(du, u)
    du = CUDA.zeros(size(du))
    u = CuArray{Float32}(u)

    return (du, u)
end

# Copy `du` and `u` to CPU (back to Float64)
function copy_to_cpu!(du, u)
    du = Array{Float64}(du)
    u = Array{Float64}(u)

    return (du, u)
end

# CUDA kernel for calculating flux value
function flux_kernel!(flux_arr, u, equations::AbstractEquations, flux::Function)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u, 1) && j <= size(u, 2) && k <= size(u, 3))
        @inbounds flux_arr[i, j, k] = flux(u[i, j, k], 1, equations)
    end

    return nothing
end

# UDA kernel for calculating weak form
function weak_form_kernel!(du, derivative_dhat, flux_arr)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        for ii in 1:size(du, 2)
            du[i, j, k] += derivative_dhat[j, ii] * flux_arr[i, ii, k]
        end
    end

    return nothing
end

# Calculate volume integral
function cuda_volume_integral!(du, u,
    mesh::TreeMesh{1},                                  # StructuredMesh{1}? 
    nonconservative_terms, equations,
    volume_integral::VolumeIntegralWeakForm,
    dg::DGSEM, cache)

    derivative_dhat = CuArray{Float32}(dg.basis.derivative_dhat)
    flux_arr = similar(u)

    flux_kernel = @cuda launch = false flux_kernel!(flux_arr, u, equations, flux)
    flux_kernel(flux_arr, u, equations, flux; configurator_3d(flux_kernel, flux_arr)...)

    weak_form_kernel = @cuda launch = false weak_form_kernel!(du, derivative_dhat, flux_arr)
    weak_form_kernel(du, derivative_dhat, flux_arr; configurator_3d(weak_form_kernel, du)...)

    return nothing
end

# CUDA kernel for prolonging two interfaces
function prolong_interfaces_kernel!(interfaces_u, u)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= 2 && j <= size(u, 1) && k <= size(u, 3))
        @inbounds interfaces_u[i, j, k] = u[j, (2-i)*size(u, 2)+(i-1)*1, (2-i)*k+(i-1)*(k%size(u, 3)+1)]
    end

    return nothing
end

# Prolong solution to interfaces
function cuda_prolong2interfaces!(cache, u,
    mesh::TreeMesh{1}, equations, surface_integral, dg::DG)

    interfaces_u = CuArray{Float32}(cache.interfaces.u)

    prolong_interfaces_kernel = @cuda launch = false prolong_interfaces_kernel!(interfaces_u, u)
    prolong_interfaces_kernel(interfaces_u, u; configurator_3d(prolong_interfaces_kernel, interfaces_u)...)

    cache.interfaces.u = interfaces_u  # Automatically copy back to CPU

    return nothing
end

# CUDA kernel for calculating surface flux value
function surface_flux_kernel!(surface_flux_arr, interfaces_u, equations::AbstractEquations{1}, surface_flux::FluxLaxFriedrichs) # Other fluxes?
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i == 1 && j <= size(interfaces_u, 2) && k <= size(interfaces_u, 3))
        @inbounds surface_flux_arr[i, j, k] = surface_flux(interfaces_u[1, j, k], interfaces_u[2, j, k], 1, equations)
    end

    return nothing
end

# CUDA kernel for setting interface flux
function interface_flux_kernel!(surface_flux_values, surface_flux_arr)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) && j <= 2 && k <= size(surface_flux_values, 3))
        @inbounds surface_flux_values[i, j, k] = surface_flux_arr[1, i,
            (j-1)*k+(2-j)*((k-1)%size(surface_flux_values, 3)+iszero(k - 1)*size(surface_flux_values, 3))]
    end

    return nothing
end

# Calculate interface fluxes
function cuda_interface_flux!(cache, mesh::TreeMesh{1},
    nonconservative_terms::False, equations, dg::DG)    # Skip `nonconservative_terms::True`

    surface_flux = dg.surface_integral.surface_flux
    interfaces_u = CuArray{Float32}(cache.interfaces.u)
    surface_flux_values = CuArray{Float32}(cache.elements.surface_flux_values)
    surface_flux_arr = CuArray{Float32}(undef, (1, size(interfaces_u, 2), size(interfaces_u, 3)))

    surface_flux_kernel = @cuda launch = false surface_flux_kernel!(surface_flux_arr, interfaces_u, equations, surface_flux)
    surface_flux_kernel(surface_flux_arr, interfaces_u, equations, surface_flux; configurator_3d(surface_flux_kernel, surface_flux_arr)...)

    interface_flux_kernel = @cuda launch = false interface_flux_kernel!(surface_flux_values, surface_flux_arr)
    interface_flux_kernel(surface_flux_values, surface_flux_arr; configurator_3d(interface_flux_kernel, surface_flux_values)...)

    cache.elements.surface_flux_values = surface_flux_values # Automatically copy back to CPU

    return nothing
end

# Prolong solution to boundaries
# Calculate boundary fluxes

# CUDA kernel for calculating surface integral
function surface_integral_kernel!(du, factor_arr, surface_flux_values)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && (j == 1 || j == size(du, 2)) && k <= size(du, 3))
        @inbounds du[i, j, k] = du[i, j, k] + (-1)^isone(j) *
                                              factor_arr[isone(j)*1+(1-isone(j))*2] *
                                              surface_flux_values[i, isone(j)*1+(1-isone(j))*2, k]
    end

    return nothing
end

# Calculate surface integrals
function cuda_surface_integral!(du, u, mesh::TreeMesh{1},           # StructuredMesh{1}? 
    equations, surface_integral, dg::DGSEM, cache)

    factor_arr = CuArray{Float32}([dg.basis.boundary_interpolation[1, 1], dg.basis.boundary_interpolation[end, 2]]) # size(u, 2) 
    surface_flux_values = CuArray{Float32}(cache.elements.surface_flux_values)

    surface_integral_kernel = @cuda launch = false surface_integral_kernel!(du, factor_arr, surface_flux_values)
    surface_integral_kernel(du, factor_arr, surface_flux_values; configurator_3d(surface_integral_kernel, du)...)

    return nothing
end

# CUDA kernel for applying inverse Jacobian 
function jacobian_kernel!(du, inverse_jacobian)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        @inbounds du[i, j, k] *= -inverse_jacobian[k]
    end

    return nothing
end

# Apply Jacobian from mapping to reference element
function cuda_jacobian!(du, mesh::TreeMesh{1},                 # StructuredMesh{1}?
    equations, dg::DG, cache)

    inverse_jacobian = CuArray{Float32}(cache.elements.inverse_jacobian)

    jacobian_kernel = @cuda launch = false jacobian_kernel!(du, inverse_jacobian)
    jacobian_kernel(du, inverse_jacobian; configurator_3d(jacobian_kernel, du)...)

    return nothing
end

#= # Calculate source terms              Overhead?
function cuda_sources!(du, u, t, source_terms::Nothing, # Skip `source_terms` has something
    equations::AbstractEquations{1}, dg::DG, cache)
    return nothing
end  =#

# Inside `rhs!()` raw implementation
#################################################################################
#= du, u = copy_to_gpu!(du, u)

cuda_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    solver.volume_integral, solver, cache)

cuda_prolong2interfaces!(
    cache, u, mesh, equations, solver.surface_integral, solver)

cuda_interface_flux!(
    cache, mesh,
    have_nonconservative_terms(equations), equations, solver)

#= cuda_prolong2boundaries!(
    cache, u, mesh, equations, solver.surface_integral, solver) =#

#= cuda_boundary_flux!(
    cache, t, boundary_conditions, mesh,
    equations, solver.surface_integral, solver) =#

cuda_surface_integral!(
    du, u, mesh, equations, solver.surface_integral, solver, cache)

cuda_jacobian!(
    du, mesh, equations, solver, cache)

#= cuda_sources!(du, u, t,
    source_terms, equations, solver, cache) =#

du, u = copy_to_cpu!(du, u) =#


node_coordinates = CuArray{Float32}(cache.elements.node_coordinates)
source_terms_arr = similar(du)

function source_terms_kernel!(source_terms_arr, u, node_coordinates, t, equations::AbstractEquations{1}, source_terms::Source)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u, 1) && j <= size(u, 2) && k <= size(u, 3))
        for ii in size(u, 1)
            u_local = xxx
        end

        @inbounds source_terms_arr[i, j, k] = source_terms(u_local, node_coordinates[1, j, k], t, equations)
    end

    return nothing
end









# For tests
#= reset_du!(du, solver, cache)

calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    solver.volume_integral, solver, cache)

prolong2interfaces!(
    cache, u, mesh, equations, solver.surface_integral, solver)

calc_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    solver.surface_integral, solver, cache)

calc_surface_integral!(
    du, u, mesh, equations, solver.surface_integral, solver, cache)

apply_jacobian!(
    du, mesh, equations, solver, cache) =#

#################################################################################

#= min(attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X), cld(length, threads))

const MAX_GRID_DIM_X = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X) # may not be used ??? =#
