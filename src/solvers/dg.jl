# Do we really need to compute the coefficients on the GPU, and do we need to
# initialize `du` and `u` with a 1D shape, as Trixi.jl does?

# Get the polynomial degree from the solver basis.
@inline polydeg(dg::DGSEM{<:LobattoLegendreBasisGPU}) = polydeg(dg.basis)

# Based on benchmarks, using `reshape` is faster than `unsafe_wrap` for GPU arrays (why?),
# but currently we are not sure about if reshape can be used for later `resize!` operations.
# But now we don't take AMR into account, so we can use `reshape` for now.

# Can we make `unsafe_wrap` faster on GPU arrays?
# Should we use `CuArray` or `AbstractGPUArray`?
@inline function wrap_array(u_ode::AbstractGPUArray, mesh::AbstractMesh, equations,
                            dg::DGSEM, cache)
    # We skip bounds checking here for better performance. 
    # @assert length(u_ode) == nvariables(equations) * nnodes(mesh) * nelements(dg, cache)

    reshape(u_ode, nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))...,
            nelements(dg, cache))
end
@inline function wrap_array_native(u_ode::AbstractGPUArray, mesh::AbstractMesh, equations,
                                   dg::DGSEM, cache)
    # We skip bounds checking here for better performance. 
    # @assert length(u_ode) == nvariables(equations) * nnodes(mesh) * nelements(dg, cache)

    unsafe_wrap(CuArray{eltype(u_ode), ndims(mesh) + 2}, pointer(u_ode),
                (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))...,
                 nelements(dg, cache)))
end

# Callbacks copy the GPU state to a host Array that already has the expected layout.
# Return it directly to support Trixi's existing callback interface.
# TODO: Keep more callback computations on the GPU and improve the interface to
# reduce host transfers and avoid pass-through adapters like this one.
@inline function wrap_array_native(u_ode::Array, mesh::TreeMesh, equations,
                                   dg::DGSEM{<:LobattoLegendreBasisGPU}, cache)
    return u_ode
end

# @inline function wrap_array(u_ode::AbstractGPUArray, mesh::AbstractMesh, equations,
#                             dg::FDSBP, cache)
#     @error("TrixiCUDA.jl does not support FDSBP yet.")
# end

# FIXME: This is a temporary workaround to avoid scalar indexing issue.
function volume_jacobian_tmp(element, mesh::TreeMesh, cache)
    inverse_jacobian = Array(cache.elements.inverse_jacobian)
    return inv(inverse_jacobian[element])^ndims(mesh)
end

# Kernel for computing the coefficients for 1D problems
function compute_coefficients_kernel!(u, node_coordinates, func::Any, t,
                                      equations::AbstractEquations{1})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2) && k <= size(u, 3))
        x_node = get_node_coords(node_coordinates, equations, j, k)

        if j == 1 # bad
            @inbounds x_node = SVector(nextfloat(x_node[1]))
        elseif j == size(u, 2) # bad
            @inbounds x_node = SVector(prevfloat(x_node[1]))
        end

        u_node = func(x_node, t, equations)

        for ii in axes(u, 1)
            @inbounds u[ii, j, k] = u_node[ii]
        end
    end

    return nothing
end

# Kernel for computing the coefficients for 2D problems
function compute_coefficients_kernel!(u, node_coordinates, func::Any, t,
                                      equations::AbstractEquations{2})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^2 && k <= size(u, 4))
        j1 = div(j - 1, size(u, 2)) + 1
        j2 = rem(j - 1, size(u, 2)) + 1

        x_node = get_node_coords(node_coordinates, equations, j1, j2, k)

        u_node = func(x_node, t, equations)

        for ii in axes(u, 1)
            @inbounds u[ii, j1, j2, k] = u_node[ii]
        end
    end

    return nothing
end

# Kernel for computing the coefficients for 3D problems
function compute_coefficients_kernel!(u, node_coordinates, func::Any, t,
                                      equations::AbstractEquations{3})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^3 && k <= size(u, 5))
        u2 = size(u, 2)

        j1 = div(j - 1, u2^2) + 1
        j2 = div(rem(j - 1, u2^2), u2) + 1
        j3 = rem(rem(j - 1, u2^2), u2) + 1

        x_node = get_node_coords(node_coordinates, equations, j1, j2, j3, k)

        u_node = func(x_node, t, equations)

        for ii in axes(u, 1)
            @inbounds u[ii, j1, j2, j3, k] = u_node[ii]
        end
    end

    return nothing
end

# Call kernels to compute the coefficients for 1D problems
function compute_coefficients_gpu(u, func, t, mesh::AbstractMesh{1}, equations, dg::DGSEM, cache)
    node_coordinates = cache.elements.node_coordinates

    # Keep captured types in generated initial conditions concrete on Julia 1.10/1.11.
    compute_coefficients_kernel = @cuda launch=false always_inline=true compute_coefficients_kernel!(u,
                                                                                                     node_coordinates,
                                                                                                     func, t,
                                                                                                     equations)
    compute_coefficients_kernel(u, node_coordinates, func, t, equations;
                                kernel_configurator_2d(compute_coefficients_kernel, size(u, 2),
                                                       size(u, 3))...)

    return u
end

# Call kernels to compute the coefficients for 2D problems
function compute_coefficients_gpu(u, func, t, mesh::AbstractMesh{2}, equations, dg::DGSEM, cache)
    node_coordinates = cache.elements.node_coordinates

    # Keep captured types in generated initial conditions concrete on Julia 1.10/1.11.
    compute_coefficients_kernel = @cuda launch=false always_inline=true compute_coefficients_kernel!(u,
                                                                                                     node_coordinates,
                                                                                                     func, t,
                                                                                                     equations)
    compute_coefficients_kernel(u, node_coordinates, func, t, equations;
                                kernel_configurator_2d(compute_coefficients_kernel, size(u, 2)^2,
                                                       size(u, 4))...)

    return u
end

# Call kernels to compute the coefficients for 3D problems
function compute_coefficients_gpu(u, func, t, mesh::AbstractMesh{3}, equations, dg::DGSEM, cache)
    node_coordinates = cache.elements.node_coordinates

    compute_coefficients_kernel = @cuda launch=false always_inline=true compute_coefficients_kernel!(u,
                                                                                                     node_coordinates,
                                                                                                     func, t,
                                                                                                     equations)
    compute_coefficients_kernel(u, node_coordinates, func, t, equations;
                                kernel_configurator_2d(compute_coefficients_kernel, size(u, 2)^3,
                                                       size(u, 5))...)

    return u
end

# These entry points mirror Trixi's names and argument order in the TrixiCUDA namespace.
# The extra CPU cache in the volume integral/RHS supports the current host-side HG indicator.

"""
    calc_volume_integral!(backend, du, u, mesh, nonconservative_terms, equations,
                          volume_integral, dg, cache_gpu, cache_cpu)

Compute the DG volume term on the GPU for weak form, flux differencing, or classic HG
shock capturing. `backend` must be a `CUDA.CUDABackend`. The additional `cache_cpu`
argument is required by the current HG indicator implementation. Writes `du` in place;
resetting `du` is fused into the first kernel.
"""
function calc_volume_integral! end

"""
    prolong2interfaces!(cache, u, mesh::TreeMesh{1}, equations, dg)
    prolong2interfaces!(backend, cache, u, mesh::TreeMesh, equations, dg)

Prolong the GPU solution to interior interfaces. As in Trixi, only the 2D/3D methods
have a backend argument. Writes `cache.interfaces.u` in place.
"""
function prolong2interfaces! end

"""
    calc_interface_flux!(surface_flux_values, mesh::TreeMesh{1}, nonconservative_terms,
                         equations, surface_integral, dg, cache)
    calc_interface_flux!(backend, surface_flux_values, mesh::TreeMesh,
                         nonconservative_terms, equations, surface_integral, dg, cache)

Compute GPU interface fluxes with the supplied surface integral and conservative/nonconservative
trait. Only 2D/3D methods take a backend. Writes into `surface_flux_values`.
"""
function calc_interface_flux! end

"""
    prolong2boundaries!(backend, cache, u, mesh, equations, dg)

Prolong the GPU solution to physical boundaries. Empty boundary containers are a no-op,
including periodic meshes. Writes `cache.boundaries.u` in place.
"""
function prolong2boundaries! end

"""
    calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
                        surface_integral, dg)

Compute GPU boundary fluxes with the supplied boundary conditions and surface integral.
Periodic boundary conditions are a no-op. The nonconservative trait is selected internally.
Writes `cache.elements.surface_flux_values` in place.
"""
function calc_boundary_flux! end

function calc_boundary_flux!(backend::CUDABackend, cache, t,
                             boundary_conditions::NamedTuple,
                             mesh::Union{TreeMesh{1}, TreeMesh{2}}, equations,
                             surface_integral, dg::DGSEM)
    return calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh,
                               have_nonconservative_terms(equations), equations,
                               surface_integral, dg)
end

"""
    prolong2mortars!(cache, u, mesh, equations, mortar_l2, dg)

Prolong the GPU solution to 2D/3D mortar interfaces with the supplied mortar operators.
As in Trixi's TreeMesh methods, no backend or cache-presence trait is passed. Empty mortar
containers are a no-op. Writes the trace buffers in `cache.mortars`.
"""
function prolong2mortars! end

"""
    calc_mortar_flux!(surface_flux_values, mesh, nonconservative_terms, equations,
                      mortar_l2, surface_integral, dg, cache)

Compute GPU mortar fluxes with the supplied mortar and surface-integral operators.
No backend or cache-presence trait is passed. Empty mortar containers are a no-op.
Writes into `surface_flux_values`.
"""
function calc_mortar_flux! end

"""
    calc_surface_integral!(backend, du, u, mesh, equations, surface_integral, dg, cache)

Accumulate GPU surface contributions in `du`. Currently supports `SurfaceIntegralWeakForm`.
The `u` argument is retained to match Trixi's signature.
"""
function calc_surface_integral! end

"""
    apply_jacobian!(backend, du, mesh, equations, dg, cache)

Apply inverse-Jacobian scaling to the GPU residual in place.
"""
function apply_jacobian! end

"""
    calc_sources!(backend, du, u, t, source_terms, equations, dg, cache)

Add source terms to the GPU residual. `source_terms = nothing` is a no-op.
"""
function calc_sources! end

"""
    rhs_hyperbolic!(du_ode, u_ode, semi::SemidiscretizationHyperbolicGPU, t)
    rhs_hyperbolic!(backend, du, u, t, mesh, equations, boundary_conditions,
                    source_terms, dg, cache_gpu, cache_cpu)

Assemble the GPU hyperbolic right-hand side from volume, interface, boundary, mortar,
surface, Jacobian, and source operations. Writes `du` in place and returns `nothing`.
The extra CPU cache supports the host-side HG indicator; 1D has no mortar stages.
"""
function rhs_hyperbolic! end
