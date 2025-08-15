# Do we really need to compute the coefficients on the GPU, and do we need to
# initialize `du` and `u` with a 1D shape, as Trixi.jl does?

# Adapt a general polynomial degree function since `DGSEMGPU` gives a `DG` type
@inline polydeg(dg::DG) = polydeg(dg.basis)

# Based on benchmarks, using `reshape` is faster than `unsafe_wrap` for GPU arrays (why?),
# but currently we are not sure about if reshape can be used for later `resize!` operations.
# But now we don't take AMR into account, so we can use `reshape` for now.

# Can we make `unsafe_wrap` faster on GPU arrays?

# Should we use `CuArray` or `AbstractGPUArray`?
@inline function wrap_array(u_ode::AbstractGPUArray, mesh::AbstractMesh, equations,
                            dg::DG, cache)
    # We skip bounds checking here for better performance. 
    # @assert length(u_ode) == nvariables(euqations) * nnodes(mesh) * nelements(dg, cache)

    reshape(u_ode, nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))...,
            nelements(dg, cache))
end

# Should we use `CuArray` or `AbstractGPUArray`?
@inline function wrap_array_native(u_ode::AbstractGPUArray, mesh::AbstractMesh, equations,
                                   dg::DG, cache)
    # We skip bounds checking here for better performance. 
    # @assert length(u_ode) == nvariables(euqations) * nnodes(mesh) * nelements(dg, cache)

    unsafe_wrap(CuArray{eltype(u_ode), ndims(mesh) + 2}, pointer(u_ode),
                (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))...,
                 nelements(dg, cache)))
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
function compute_coefficients_gpu(u, func, t, mesh::AbstractMesh{1}, equations, dg::DG, cache)
    node_coordinates = cache.elements.node_coordinates

    compute_coefficients_kernel = @cuda launch=false compute_coefficients_kernel!(u,
                                                                                  node_coordinates,
                                                                                  func, t,
                                                                                  equations)
    compute_coefficients_kernel(u, node_coordinates, func, t, equations;
                                kernel_configurator_2d(compute_coefficients_kernel, size(u, 2),
                                                       size(u, 3))...)

    return u
end

# Call kernels to compute the coefficients for 2D problems
function compute_coefficients_gpu(u, func, t, mesh::AbstractMesh{2}, equations, dg::DG, cache)
    node_coordinates = cache.elements.node_coordinates

    compute_coefficients_kernel = @cuda launch=false compute_coefficients_kernel!(u,
                                                                                  node_coordinates,
                                                                                  func, t,
                                                                                  equations)
    compute_coefficients_kernel(u, node_coordinates, func, t, equations;
                                kernel_configurator_2d(compute_coefficients_kernel, size(u, 2)^2,
                                                       size(u, 4))...)

    return u
end

# Call kernels to compute the coefficients for 3D problems
function compute_coefficients_gpu(u, func, t, mesh::AbstractMesh{3}, equations, dg::DG, cache)
    node_coordinates = cache.elements.node_coordinates

    compute_coefficients_kernel = @cuda launch=false compute_coefficients_kernel!(u,
                                                                                  node_coordinates,
                                                                                  func, t,
                                                                                  equations)
    compute_coefficients_kernel(u, node_coordinates, func, t, equations;
                                kernel_configurator_2d(compute_coefficients_kernel, size(u, 2)^3,
                                                       size(u, 5))...)

    return u
end

# Generic GPU kernels definitions and docs. 
# More specific function definitions and comments are in dg_1d.jl, dg_2d.jl, dg_3d.jl files.
"""
    cuda_volume_integral!(du, u, mesh, nonconservative_terms, equations, volume_integral, dg,
                          cache_gpu, cache_cpu)

Compute the DG volume term on the GPU. This routine dispatches to the appropriate kernel 
implementation depending on `volume_integral` and whether `nonconservative_terms` are present.

Variants include:
  - [`VolumeIntegralWeakForm`](https://trixi-framework.github.io/TrixiDocumentation/stable/reference-trixi/#Trixi.VolumeIntegralWeakForm) 
    (classical weak form);
  - [`VolumeIntegralFluxDifferencing`](https://trixi-framework.github.io/TrixiDocumentation/stable/reference-trixi/#Trixi.VolumeIntegralFluxDifferencing) 
    (split form using symmetric two-point flux) with and without non-conservative terms;
  - [`VolumeIntegralShockCapturingHG`](https://trixi-framework.github.io/TrixiDocumentation/stable/reference-trixi/#Trixi.VolumeIntegralShockCapturingHG) 
    (hybrid DG–FV scheme using an indicator) with and without non-conservative terms.

Writes into `du` in place. Returns `nothing`.
"""
function cuda_volume_integral!(du, u, mesh, nonconservative_terms, equations, volume_integral, dg,
                               cache_gpu, cache_cpu)
end

"""
    cuda_prolong2interfaces!(u, mesh, equations, cache)

Prolong the solution from element interiors to interior interfaces on the GPU.
This prepares trace data used by interface flux kernels.

Writes interface traces to `cache.interfaces.u`. Returns `nothing`.
"""
function cuda_prolong2interfaces!(u, mesh, equations, cache) end

"""
    cuda_interface_flux!(mesh, nonconservative_terms, equations, dg, cache)

Compute numerical fluxes on interior interfaces on the GPU, with or without
non-conservative terms.

If `nonconservative_terms` is `False`, use the conservative surface flux from 
`dg.surface_integral.surface_flux`; if `nonconservative_terms` is `True`, take 
both the conservative and non-conservative fluxes from it and accumulate both.

Writes interface fluxes to `cache.elements.surface_flux_values`. Returns `nothing`.
"""
function cuda_interface_flux!(mesh, nonconservative_terms, equations, dg, cache) end

"""
    cuda_prolong2boundaries!(u, mesh, boundary_conditions, equations, cache)

Prolong the solution from element interiors to physical boundaries on the GPU.
For periodic boundaries this is a no-op method.

Variants include:
  - [`BoundaryConditionPeriodic`](https://trixi-framework.github.io/TrixiDocumentation/stable/reference-trixi/#Trixi.boundary_condition_periodic) 
    (no operation for periodic boundaries);
  - `NamedTuple` (boundary conditions provided as a `NamedTuple`).

Writes boundary traces to `cache.boundaries.u` when applicable. Returns `nothing`.
"""
function cuda_prolong2boundaries!(u, mesh, boundary_conditions, equations, cache) end

"""
    cuda_boundary_flux!(t, mesh, boundary_conditions, nonconservative_terms, equations, dg, 
                        cache)

Compute numerical fluxes on physical boundaries on the GPU at time `t`. The
flux implementation is taken from `dg.surface_integral.surface_flux` and may
handle non-conservative contributions when requested. For periodic boundaries
this is a no-op method.

Variants include:
  - [`BoundaryConditionPeriodic`](https://trixi-framework.github.io/TrixiDocumentation/stable/reference-trixi/#Trixi.boundary_condition_periodic) 
    (no operation for periodic boundaries);
  - `NamedTuple` (boundary conditions provided as a `NamedTuple`) with and without non-conservative terms.

Writes boundary flux contributions to `cache.elements.surface_flux_values`. 
Returns `nothing`.
"""
function cuda_boundary_flux!(t, mesh, boundary_conditions, nonconservative_terms, equations,
                             dg, cache)
end

"""
    cuda_prolong2mortars!(u, mesh, cache_mortars, dg, cache)

Prolong the solution from element faces to mortar interfaces on the GPU for 2D/3D problems.

When `cache_mortars` is `True`, fills the mortar trace buffers in `cache.mortars` using the
appropriate transfer operators from `dg.mortar`. When `cache_mortars` is `False`, this is a no-op.

Writes mortar trace buffers to `cache.mortars` (layout depends on dimension).
Returns `nothing`.
"""
function cuda_prolong2mortars!(u, mesh, cache_mortars, dg, cache) end

"""
    cuda_mortar_flux!(mesh, cache_mortars, nonconservative_terms, equations, dg, cache)

Compute numerical fluxes on mortar interfaces on the GPU for 2D/3D problems.

Uses `dg.surface_integral.surface_flux`; when `nonconservative_terms` is `True`,
also includes the corresponding non-conservative partner flux. Contributions are
assembled back to the element side via the reverse mortar maps. When
`cache_mortars` is `False`, this is a no-op.

Writes to `cache.elements.surface_flux_values` (via temporary mortar buffers).
Returns `nothing`.
"""
function cuda_mortar_flux!(mesh, cache_mortars, nonconservative_terms, equations, dg, cache) end

"""
    cuda_surface_integral!(du, mesh, equations, dg, cache)

Accumulate interface and boundary fluxes into the element residual on the GPU
using the DG surface integral with boundary interpolation weights.

Updates `du` in place. Returns `nothing`.
"""
function cuda_surface_integral!(du, mesh, equations, dg, cache) end

"""
    cuda_jacobian!(du, mesh, equations, cache)

Apply the inverse Jacobian and geometric scaling on the GPU to map reference
operators to physical space as part of the semidiscretization.

Updates `du` in place. Returns `nothing`.
"""
function cuda_jacobian!(du, mesh, equations, cache) end

"""
    cuda_sources!(du, u, t, source_terms, equations, cache)

Evaluate and add source terms on the GPU at time `t`. When `source_terms` is not presented,
this method is a no-op.

Updates `du` in place. Returns `nothing`.
"""
function cuda_sources!(du, u, t, source_terms, equations, cache) end

"""
    rhs_gpu!(du, u, t, mesh, equations, boundary_conditions, source_terms, dg, cache_gpu, 
             cache_cpu)

Assemble the semidiscrete right-hand side on the GPU. This routine orchestrates
CUDA kernels for: volume integrals; prolongation to interior interfaces (and, in
2D/3D, to mortars with the associated mortar fluxes); interior-interface fluxes;
boundary prolongation and boundary fluxes; surface accumulation; inverse-Jacobian
scaling; and source terms.

Variants:
  - [`TreeMesh{1}`](https://trixi-framework.github.io/TrixiDocumentation/stable/reference-trixi/#Trixi.TreeMesh): 
    no mortar stages.
  - [`TreeMesh{2}`](https://trixi-framework.github.io/TrixiDocumentation/stable/reference-trixi/#Trixi.TreeMesh) / 
    [`TreeMesh{3}`](https://trixi-framework.github.io/TrixiDocumentation/stable/reference-trixi/#Trixi.TreeMesh): 
    includes [`cuda_prolong2mortars!`](@ref) and [`cuda_mortar_flux!`](@ref).

Effects: writes results into `du` in place. Return value: `nothing`.

Notes: `reset_du!(du)` is fused into the first kernel and therefore not called separately.
"""
function rhs_gpu!(du, u, t, mesh, equations, boundary_conditions, source_terms, dg,
                  cache_gpu, cache_cpu)
end
