#include("header.jl") # Remove it after first run to avoid recompilation

# Set random seed
Random.seed!(12345)

# The header part of test
equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test
solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs)

coordinates_min = 0.0
coordinates_max = 2.0
mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=4,
    n_cells_max=10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
    source_terms=source_terms_convergence_test)

# Unpack to get key elements
@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

# Create pesudo `u`, `du` and `t` for tests
t = 0.0
l = nvariables(equations) * nnodes(solver)^ndims(mesh) * nelements(solver, cache)
u_ode = rand(Float64, l)
du_ode = rand(Float64, l)
u = wrap_array(u_ode, mesh, equations, solver, cache)
du = wrap_array(du_ode, mesh, equations, solver, cache)

# CUDA kernel configurators for 1D, 2D, and 3D arrays
#################################################################################

# CUDA kernel configurator for 1D array computing
function configurator_1d(kernel::CUDA.HostKernel, array::CuArray{Float32,1})
    config = launch_configuration(kernel.fun)

    threads = min(length(array), config.threads)
    blocks = cld(length(array), threads)

    return (threads=threads, blocks=blocks)
end

# CUDA kernel configurator for 2D array computing
function configurator_2d(kernel::CUDA.HostKernel, array::CuArray{Float32,2})
    config = launch_configuration(kernel.fun)

    threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 2))), 2))
    blocks = map(cld, size(array), threads)

    return (threads=threads, blocks=blocks)
end

# CUDA kernel configurator for 3D array computing
function configurator_3d(kernel::CUDA.HostKernel, array::CuArray{Float32,3})
    config = launch_configuration(kernel.fun)

    threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 3))), 3))
    blocks = map(cld, size(array), threads)

    return (threads=threads, blocks=blocks)
end

# Rewrite `rhs!()` from `trixi/src/solvers/dgsem_tree/dg_1d.jl`
#################################################################################

# Copy data to GPU (run as Float32)
function copy_to_gpu!(du, u)
    du = CUDA.zeros(size(du))
    u = CuArray{Float32}(u)

    return (du, u)
end

# Copy data to CPU (back to Float64)
function copy_to_cpu!(du, u)
    du = Array{Float64}(du)
    u = Array{Float64}(u)

    return (du, u)
end

# Rewrite `get_node_vars()` as a helper function
@inline function get_nodes_helper(u, equations, indices...)

    SVector(ntuple(@inline(v -> u[v, indices...]), Val(nvariables(equations))))
end

# CUDA kernel for calculating fluxes along normal direction 1 
function flux_kernel!(flux_arr, u, equations::AbstractEquations{1}, flux::Function)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(u, 1) && j <= size(u, 2) && k <= size(u, 3))
        u_node = get_nodes_helper(u, equations, j, k)
        @inbounds flux_arr[i, j, k] = flux(u_node, 1, equations)[i]
    end

    return nothing
end

# CUDA kernel for calculating weak form
function weak_form_kernel!(du, derivative_dhat, flux_arr)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        @inbounds begin
            for ii in 1:size(du, 2)
                du[i, j, k] += derivative_dhat[j, ii] * flux_arr[i, ii, k]
            end
        end
    end

    return nothing
end

# Calculate volume integral
function cuda_volume_integral!(du, u, mesh::TreeMesh{1},
    nonconservative_terms, equations,
    volume_integral::VolumeIntegralWeakForm, dg::DGSEM)

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
function cuda_prolong2interfaces!(cache, u, mesh::TreeMesh{1})

    interfaces_u = CuArray{Float32}(cache.interfaces.u)

    prolong_interfaces_kernel = @cuda launch = false prolong_interfaces_kernel!(interfaces_u, u)
    prolong_interfaces_kernel(interfaces_u, u; configurator_3d(prolong_interfaces_kernel, interfaces_u)...)

    cache.interfaces.u = interfaces_u  # Automatically copy back to CPU

    return nothing
end

# Rewrite `get_surface_node_vars()` as a helper function
@inline function get_surface_node_vars(u, equations, indices...)

    u_ll = SVector(ntuple(@inline(v -> u[1, v, indices...]), Val(nvariables(equations))))
    u_rr = SVector(ntuple(@inline(v -> u[2, v, indices...]), Val(nvariables(equations))))

    return u_ll, u_rr
end

# CUDA kernel for calculating surface fluxes 
function surface_flux_kernel!(surface_flux_arr, interfaces_u, equations::AbstractEquations{1}, surface_flux::FluxLaxFriedrichs) # ::Any?
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i == 1 && j <= size(interfaces_u, 2) && k <= size(interfaces_u, 3))
        u_ll, u_rr = get_surface_node_vars(u, equations, interface)
        @inbounds surface_flux_arr[i, j, k] = surface_flux(interfaces_u[1, j, k], interfaces_u[2, j, k], 1, equations)
    end

    return nothing
end

# CUDA kernel for setting interface fluxes
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
    nonconservative_terms::False, equations, dg::DG)                # Need ...? `nonconservative_terms::True`

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

# Inside `rhs!()` raw implementation
#################################################################################
#= du, u = copy_to_gpu!(du, u)

cuda_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    solver.volume_integral, solver)

cuda_prolong2interfaces!(cache, u, mesh) =#


# For tests
reset_du!(du, solver, cache)

calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    solver.volume_integral, solver, cache)

prolong2interfaces!(
    cache, u, mesh, equations, solver.surface_integral, solver)

#= calc_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    solver.surface_integral, solver, cache)

calc_surface_integral!(
    du, u, mesh, equations, solver.surface_integral, solver, cache)

apply_jacobian!(
    du, mesh, equations, solver, cache) =#