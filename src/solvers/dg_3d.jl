# Everything related to a DG semidiscretization in 3D.

include("dg_3d_kernel.jl")

# Functions that begin with `cuda_` are the functions that pack CUDA kernels together to do 
# partial work in semidiscretization. They are used to invoke kernels from the host (i.e., CPU) 
# and run them on the device (i.e., GPU).

# Note that `volume_integral::VolumeIntegralPureLGLFiniteVolume` is currently experimental
# in Trixi.jl and it is not implemented here.

# The maximum number of threads per block is the dominant factor when choosing the optimization 
# method. But note that there are other factors such as max register number per block and we will
# enhance the checking mechanism in the future.

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::DG,
                               cache_gpu, cache_cpu)
    RealT = eltype(du)

    derivative_dhat = dg.basis.derivative_dhat

    thread_per_block = size(du, 1) * size(du, 2)^3
    shmem_per_block = (size(du, 2)^2 + size(du, 1) * 3 * size(du, 2)^3) * sizeof(RealT)
    if thread_per_block <= MAX_THREADS_PER_BLOCK && shmem_per_block <= MAX_SHARED_MEMORY_PER_BLOCK
        # Go with the optimized version (frequent use) 
        threads = (size(du, 1), size(du, 2)^3, 1)
        blocks = (1, 1, size(du, 5))
        @cuda threads=threads blocks=blocks shmem=shmem_per_block flux_weak_form_kernel!(du, u,
                                                                                         derivative_dhat,
                                                                                         equations,
                                                                                         flux)
    else
        # How to optimize when size is large (less common use)?
        flux_arr1 = similar(u)
        flux_arr2 = similar(u)
        flux_arr3 = similar(u)

        flux_kernel = @cuda launch=false flux_kernel!(flux_arr1, flux_arr2, flux_arr3, u, equations, flux)
        flux_kernel(flux_arr1, flux_arr2, flux_arr3, u, equations, flux;
                    kernel_configurator_2d(flux_kernel, size(u, 2)^3, size(u, 5))...)

        weak_form_kernel = @cuda launch=false weak_form_kernel!(du, derivative_dhat, flux_arr1,
                                                                flux_arr2, flux_arr3)
        weak_form_kernel(du, derivative_dhat, flux_arr1, flux_arr2, flux_arr3;
                         kernel_configurator_3d(weak_form_kernel, size(du, 1), size(du, 2)^3,
                                                size(du, 5))...)
    end

    return nothing
end

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralFluxDifferencing, dg::DG,
                               cache_gpu, cache_cpu)
    RealT = eltype(du)

    volume_flux = volume_integral.volume_flux
    derivative_split = dg.basis.derivative_split

    thread_per_block = size(du, 2)^3
    shmem_per_block = (size(du, 2)^2 + size(du, 1) * size(du, 2)^3) * sizeof(RealT)
    if thread_per_block <= MAX_THREADS_PER_BLOCK && shmem_per_block <= MAX_SHARED_MEMORY_PER_BLOCK
        # Go with the optimized version (frequent use)
        threads = (1, size(du, 2)^3, 1)
        blocks = (1, 1, size(du, 5))
        @cuda threads=threads blocks=blocks shmem=shmem_per_block volume_flux_integral_kernel!(du, u,
                                                                                               derivative_split,
                                                                                               equations,
                                                                                               volume_flux)
    else
        # How to optimize when size is large (less common use)?
        volume_flux_arr1 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                          size(u, 2), size(u, 5))
        volume_flux_arr2 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                          size(u, 2), size(u, 5))
        volume_flux_arr3 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                          size(u, 2), size(u, 5))

        volume_flux_kernel = @cuda launch=false volume_flux_kernel!(volume_flux_arr1, volume_flux_arr2,
                                                                    volume_flux_arr3, u, equations,
                                                                    volume_flux)
        volume_flux_kernel(volume_flux_arr1, volume_flux_arr2, volume_flux_arr3, u, equations,
                           volume_flux;
                           kernel_configurator_2d(volume_flux_kernel, size(u, 2)^4, size(u, 5))...)

        volume_integral_kernel = @cuda launch=false volume_integral_kernel!(du, derivative_split,
                                                                            volume_flux_arr1,
                                                                            volume_flux_arr2,
                                                                            volume_flux_arr3, equations)
        volume_integral_kernel(du, derivative_split, volume_flux_arr1, volume_flux_arr2,
                               volume_flux_arr3, equations;
                               kernel_configurator_3d(volume_integral_kernel, size(du, 1),
                                                      size(du, 2)^3, size(du, 5))...)
    end

    return nothing
end

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms::True, equations,
                               volume_integral::VolumeIntegralFluxDifferencing, dg::DG,
                               cache_gpu, cache_cpu)
    RealT = eltype(du)

    symmetric_flux, nonconservative_flux = dg.volume_integral.volume_flux
    derivative_split = dg.basis.derivative_split

    thread_per_block = size(du, 2)^3
    shmem_per_block = (size(du, 2)^2 + size(du, 1) * size(du, 2)^3) * sizeof(RealT)
    if thread_per_block <= MAX_THREADS_PER_BLOCK && shmem_per_block <= MAX_SHARED_MEMORY_PER_BLOCK
        # Go with the optimized version (frequent use)
        threads = (1, size(du, 2)^3, 1)
        blocks = (1, 1, size(du, 5))
        @cuda threads=threads blocks=blocks shmem=shmem_per_block volume_flux_integral_kernel!(du, u,
                                                                                               derivative_split,
                                                                                               equations,
                                                                                               symmetric_flux,
                                                                                               nonconservative_flux)
    else
        # How to optimize when size is large (less common use)?
        symmetric_flux_arr1 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                             size(u, 2), size(u, 5))
        symmetric_flux_arr2 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                             size(u, 2), size(u, 5))
        symmetric_flux_arr3 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                             size(u, 2), size(u, 5))
        noncons_flux_arr1 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                           size(u, 2), size(u, 5))
        noncons_flux_arr2 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                           size(u, 2), size(u, 5))
        noncons_flux_arr3 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                           size(u, 2), size(u, 5))

        noncons_volume_flux_kernel = @cuda launch=false noncons_volume_flux_kernel!(symmetric_flux_arr1,
                                                                                    symmetric_flux_arr2,
                                                                                    symmetric_flux_arr3,
                                                                                    noncons_flux_arr1,
                                                                                    noncons_flux_arr2,
                                                                                    noncons_flux_arr3, u,
                                                                                    derivative_split,
                                                                                    equations,
                                                                                    symmetric_flux,
                                                                                    nonconservative_flux)
        noncons_volume_flux_kernel(symmetric_flux_arr1, symmetric_flux_arr2, symmetric_flux_arr3,
                                   noncons_flux_arr1, noncons_flux_arr2, noncons_flux_arr3, u,
                                   derivative_split, equations, symmetric_flux, nonconservative_flux;
                                   kernel_configurator_2d(noncons_volume_flux_kernel,
                                                          size(u, 2)^4, size(u, 5))...)

        volume_integral_kernel = @cuda launch=false volume_integral_kernel!(du, derivative_split,
                                                                            symmetric_flux_arr1,
                                                                            symmetric_flux_arr2,
                                                                            symmetric_flux_arr3,
                                                                            noncons_flux_arr1,
                                                                            noncons_flux_arr2,
                                                                            noncons_flux_arr3)
        volume_integral_kernel(du, derivative_split, symmetric_flux_arr1, symmetric_flux_arr2,
                               symmetric_flux_arr3, noncons_flux_arr1, noncons_flux_arr2,
                               noncons_flux_arr3;
                               kernel_configurator_3d(volume_integral_kernel, size(du, 1),
                                                      size(du, 2)^3, size(du, 5))...)
    end

    return nothing
end

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralShockCapturingHG, dg::DG,
                               cache_gpu, cache_cpu)
    RealT = eltype(du)

    volume_flux_dg, volume_flux_fv = dg.volume_integral.volume_flux_dg,
                                     dg.volume_integral.volume_flux_fv
    indicator = dg.volume_integral.indicator
    derivative_split = dg.basis.derivative_split
    inverse_weights = dg.basis.inverse_weights

    # TODO: Get copies of `u` and `du` on both device and host
    alpha = indicator(Array(u), mesh, equations, dg, cache_cpu)
    alpha = CuArray(alpha)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))

    thread_per_block = size(du, 2)^3
    shmem_per_block = (size(u, 2)^2 + size(u, 1) * size(u, 2)^2 * (size(u, 2) + 1) * 3 +
                       size(u, 1) * size(u, 2)^3) * sizeof(RealT)
    if thread_per_block <= MAX_THREADS_PER_BLOCK && shmem_per_block <= MAX_SHARED_MEMORY_PER_BLOCK
        # Go with the optimized version (frequent use)
        threads = (1, size(u, 2)^3, 1)
        blocks = (1, 1, size(u, 5))
        @cuda threads=threads blocks=blocks shmem=shmem_per_block volume_flux_integral_dgfv_kernel!(du, u, alpha, atol,
                                                                                                    derivative_split,
                                                                                                    inverse_weights,
                                                                                                    equations,
                                                                                                    volume_flux_dg,
                                                                                                    volume_flux_fv)
    else
        # TODO: Remove `fstar` from cache initialization
        fstar1_L = cache_gpu.fstar1_L
        fstar1_R = cache_gpu.fstar1_R
        fstar2_L = cache_gpu.fstar2_L
        fstar2_R = cache_gpu.fstar2_R
        fstar3_L = cache_gpu.fstar3_L
        fstar3_R = cache_gpu.fstar3_R

        volume_flux_arr1 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                          size(u, 2), size(u, 5))
        volume_flux_arr2 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                          size(u, 2), size(u, 5))
        volume_flux_arr3 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                          size(u, 2), size(u, 5))

        volume_flux_dgfv_kernel = @cuda launch=false volume_flux_dgfv_kernel!(volume_flux_arr1,
                                                                              volume_flux_arr2,
                                                                              volume_flux_arr3,
                                                                              fstar1_L,
                                                                              fstar1_R, fstar2_L,
                                                                              fstar2_R,
                                                                              fstar3_L, fstar3_R, u,
                                                                              alpha, atol,
                                                                              equations,
                                                                              volume_flux_dg,
                                                                              volume_flux_fv)
        volume_flux_dgfv_kernel(volume_flux_arr1, volume_flux_arr2, volume_flux_arr3, fstar1_L,
                                fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R, u, alpha, atol,
                                equations, volume_flux_dg, volume_flux_fv;
                                kernel_configurator_2d(volume_flux_dgfv_kernel, size(u, 2)^4,
                                                       size(u, 5))...)

        volume_integral_dgfv_kernel = @cuda launch=false volume_integral_dgfv_kernel!(du, alpha,
                                                                                      derivative_split,
                                                                                      inverse_weights,
                                                                                      volume_flux_arr1,
                                                                                      volume_flux_arr2,
                                                                                      volume_flux_arr3,
                                                                                      fstar1_L, fstar1_R,
                                                                                      fstar2_L, fstar2_R,
                                                                                      fstar3_L, fstar3_R,
                                                                                      atol, equations)
        volume_integral_dgfv_kernel(du, alpha, derivative_split, inverse_weights, volume_flux_arr1,
                                    volume_flux_arr2, volume_flux_arr3, fstar1_L, fstar1_R,
                                    fstar2_L, fstar2_R, fstar3_L, fstar3_R, atol, equations;
                                    kernel_configurator_3d(volume_integral_dgfv_kernel, size(du, 1),
                                                           size(du, 2)^3, size(du, 5))...)
    end

    return nothing
end

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{3}, nonconservative_terms::True, equations,
                               volume_integral::VolumeIntegralShockCapturingHG, dg::DG,
                               cache_gpu, cache_cpu)
    RealT = eltype(du)

    volume_flux_dg, noncons_flux_dg = dg.volume_integral.volume_flux_dg
    volume_flux_fv, noncons_flux_fv = dg.volume_integral.volume_flux_fv
    indicator = dg.volume_integral.indicator
    derivative_split = dg.basis.derivative_split
    inverse_weights = dg.basis.inverse_weights

    # TODO: Get copies of `u` and `du` on both device and host
    alpha = indicator(Array(u), mesh, equations, dg, cache_cpu)
    alpha = CuArray(alpha)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))

    thread_per_block = size(du, 2)^3
    shmem_per_block = (size(u, 2)^2 + size(u, 1) * size(u, 2)^2 * (size(u, 2) + 1) * 6 +
                       size(u, 1) * size(u, 2)^3) * sizeof(RealT)
    if thread_per_block <= MAX_THREADS_PER_BLOCK && shmem_per_block <= MAX_SHARED_MEMORY_PER_BLOCK
        # Go with the optimized version (frequent use)
        threads = (1, size(u, 2)^3, 1)
        blocks = (1, 1, size(u, 5))
        @cuda threads=threads blocks=blocks shmem=shmem_per_block volume_flux_integral_dgfv_kernel!(du, u, alpha, atol,
                                                                                                    derivative_split,
                                                                                                    inverse_weights,
                                                                                                    equations,
                                                                                                    volume_flux_dg,
                                                                                                    noncons_flux_dg,
                                                                                                    volume_flux_fv,
                                                                                                    noncons_flux_fv)
    else
        # TODO: Remove `fstar` from cache initialization
        fstar1_L = cache_gpu.fstar1_L
        fstar1_R = cache_gpu.fstar1_R
        fstar2_L = cache_gpu.fstar2_L
        fstar2_R = cache_gpu.fstar2_R
        fstar3_L = cache_gpu.fstar3_L
        fstar3_R = cache_gpu.fstar3_R

        volume_flux_arr1 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                          size(u, 2), size(u, 5))
        volume_flux_arr2 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                          size(u, 2), size(u, 5))
        volume_flux_arr3 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                          size(u, 2), size(u, 5))
        noncons_flux_arr1 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                           size(u, 2), size(u, 5))
        noncons_flux_arr2 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                           size(u, 2), size(u, 5))
        noncons_flux_arr3 = CuArray{RealT}(undef, size(u, 1), size(u, 2), size(u, 2), size(u, 2),
                                           size(u, 2), size(u, 5))

        volume_flux_dgfv_kernel = @cuda launch=false volume_flux_dgfv_kernel!(volume_flux_arr1,
                                                                              volume_flux_arr2,
                                                                              volume_flux_arr3,
                                                                              noncons_flux_arr1,
                                                                              noncons_flux_arr2,
                                                                              noncons_flux_arr3,
                                                                              fstar1_L, fstar1_R,
                                                                              fstar2_L, fstar2_R,
                                                                              fstar3_L, fstar3_R,
                                                                              u, alpha, atol,
                                                                              derivative_split,
                                                                              equations,
                                                                              volume_flux_dg,
                                                                              noncons_flux_dg,
                                                                              volume_flux_fv,
                                                                              noncons_flux_fv)
        volume_flux_dgfv_kernel(volume_flux_arr1, volume_flux_arr2, volume_flux_arr3,
                                noncons_flux_arr1, noncons_flux_arr2, noncons_flux_arr3,
                                fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R,
                                u, alpha, atol, derivative_split, equations, volume_flux_dg,
                                noncons_flux_dg, volume_flux_fv, noncons_flux_fv;
                                kernel_configurator_2d(volume_flux_dgfv_kernel, size(u, 2)^4,
                                                       size(u, 5))...)

        volume_integral_dgfv_kernel = @cuda launch=false volume_integral_dgfv_kernel!(du, alpha,
                                                                                      derivative_split,
                                                                                      inverse_weights,
                                                                                      volume_flux_arr1,
                                                                                      volume_flux_arr2,
                                                                                      volume_flux_arr3,
                                                                                      noncons_flux_arr1,
                                                                                      noncons_flux_arr2,
                                                                                      noncons_flux_arr3,
                                                                                      fstar1_L, fstar1_R,
                                                                                      fstar2_L, fstar2_R,
                                                                                      fstar3_L, fstar3_R,
                                                                                      atol, equations)
        volume_integral_dgfv_kernel(du, alpha, derivative_split, inverse_weights, volume_flux_arr1,
                                    volume_flux_arr2, volume_flux_arr3, noncons_flux_arr1,
                                    noncons_flux_arr2, noncons_flux_arr3, fstar1_L, fstar1_R,
                                    fstar2_L, fstar2_R, fstar3_L, fstar3_R, atol, equations;
                                    kernel_configurator_3d(volume_integral_dgfv_kernel, size(du, 1),
                                                           size(du, 2)^3, size(du, 5))...)
    end

    return nothing
end

# Pack kernels to prolonging solution to interfaces
function cuda_prolong2interfaces!(u, mesh::TreeMesh{3}, equations, cache)
    neighbor_ids = cache.interfaces.neighbor_ids
    orientations = cache.interfaces.orientations
    interfaces_u = cache.interfaces.u

    prolong_interfaces_kernel = @cuda launch=false prolong_interfaces_kernel!(interfaces_u, u,
                                                                              neighbor_ids,
                                                                              orientations,
                                                                              equations)
    prolong_interfaces_kernel(interfaces_u, u, neighbor_ids, orientations, equations;
                              kernel_configurator_2d(prolong_interfaces_kernel,
                                                     size(interfaces_u, 2) *
                                                     size(interfaces_u, 3)^2,
                                                     size(interfaces_u, 5))...)

    return nothing
end

# Pack kernels for calculating interface fluxes
function cuda_interface_flux!(mesh::TreeMesh{3}, nonconservative_terms::False, equations, dg::DG,
                              cache)
    RealT = eltype(cache.elements)

    surface_flux = dg.surface_integral.surface_flux

    neighbor_ids = cache.interfaces.neighbor_ids
    orientations = cache.interfaces.orientations
    interfaces_u = cache.interfaces.u
    surface_flux_values = cache.elements.surface_flux_values
    surface_flux_arr = CuArray{RealT}(undef, size(interfaces_u)[2:end]...)

    surface_flux_kernel = @cuda launch=false surface_flux_kernel!(surface_flux_arr, interfaces_u,
                                                                  orientations, equations,
                                                                  surface_flux)
    surface_flux_kernel(surface_flux_arr, interfaces_u, orientations, equations, surface_flux;
                        kernel_configurator_3d(surface_flux_kernel, size(interfaces_u, 3),
                                               size(interfaces_u, 4),
                                               size(interfaces_u, 5))...)

    interface_flux_kernel = @cuda launch=false interface_flux_kernel!(surface_flux_values,
                                                                      surface_flux_arr,
                                                                      neighbor_ids, orientations,
                                                                      equations)
    interface_flux_kernel(surface_flux_values, surface_flux_arr, neighbor_ids, orientations,
                          equations;
                          kernel_configurator_3d(interface_flux_kernel,
                                                 size(surface_flux_values, 1),
                                                 size(interfaces_u, 3)^2,
                                                 size(interfaces_u, 5))...)

    return nothing
end

# Pack kernels for calculating interface fluxes
function cuda_interface_flux!(mesh::TreeMesh{3}, nonconservative_terms::True, equations, dg::DG,
                              cache)
    RealT = eltype(cache.elements)

    surface_flux, nonconservative_flux = dg.surface_integral.surface_flux

    neighbor_ids = cache.interfaces.neighbor_ids
    orientations = cache.interfaces.orientations
    interfaces_u = cache.interfaces.u
    surface_flux_values = cache.elements.surface_flux_values

    surface_flux_arr = CuArray{RealT}(undef, size(interfaces_u)[2:end]...)
    noncons_left_arr = CuArray{RealT}(undef, size(interfaces_u)[2:end]...)
    noncons_right_arr = CuArray{RealT}(undef, size(interfaces_u)[2:end]...)

    surface_noncons_flux_kernel = @cuda launch=false surface_noncons_flux_kernel!(surface_flux_arr,
                                                                                  noncons_left_arr,
                                                                                  noncons_right_arr,
                                                                                  interfaces_u,
                                                                                  orientations,
                                                                                  equations,
                                                                                  surface_flux,
                                                                                  nonconservative_flux)
    surface_noncons_flux_kernel(surface_flux_arr, noncons_left_arr, noncons_right_arr, interfaces_u,
                                orientations, equations, surface_flux, nonconservative_flux;
                                kernel_configurator_3d(surface_noncons_flux_kernel,
                                                       size(interfaces_u, 3),
                                                       size(interfaces_u, 4),
                                                       size(interfaces_u, 5))...)

    interface_flux_kernel = @cuda launch=false interface_flux_kernel!(surface_flux_values,
                                                                      surface_flux_arr,
                                                                      noncons_left_arr,
                                                                      noncons_right_arr,
                                                                      neighbor_ids, orientations,
                                                                      equations)
    interface_flux_kernel(surface_flux_values, surface_flux_arr, noncons_left_arr,
                          noncons_right_arr,
                          neighbor_ids, orientations, equations;
                          kernel_configurator_3d(interface_flux_kernel,
                                                 size(surface_flux_values, 1),
                                                 size(interfaces_u, 3)^2,
                                                 size(interfaces_u, 5))...)

    return nothing
end

# Dummy function for asserting boundaries
function cuda_prolong2boundaries!(u, mesh::TreeMesh{3},
                                  boundary_condition::BoundaryConditionPeriodic, equations, cache)
    @assert iszero(length(cache.boundaries.orientations))
end

# Pack kernels for prolonging solution to boundaries
function cuda_prolong2boundaries!(u, mesh::TreeMesh{3}, boundary_conditions::NamedTuple, equations,
                                  cache)
    neighbor_ids = cache.boundaries.neighbor_ids
    neighbor_sides = cache.boundaries.neighbor_sides
    orientations = cache.boundaries.orientations
    boundaries_u = cache.boundaries.u

    prolong_boundaries_kernel = @cuda launch=false prolong_boundaries_kernel!(boundaries_u, u,
                                                                              neighbor_ids,
                                                                              neighbor_sides,
                                                                              orientations,
                                                                              equations)
    prolong_boundaries_kernel(boundaries_u, u, neighbor_ids, neighbor_sides, orientations,
                              equations;
                              kernel_configurator_2d(prolong_boundaries_kernel,
                                                     size(boundaries_u, 2) *
                                                     size(boundaries_u, 3)^2,
                                                     size(boundaries_u, 5))...)

    return nothing
end

# Dummy function for asserting boundary fluxes
function cuda_boundary_flux!(t, mesh::TreeMesh{3}, boundary_condition::BoundaryConditionPeriodic,
                             nonconservative_terms, equations, dg::DG, cache)
    @assert iszero(length(cache.boundaries.orientations))
end

# Pack kernels for calculating boundary fluxes
function cuda_boundary_flux!(t, mesh::TreeMesh{3}, boundary_conditions::NamedTuple,
                             nonconservative_terms, equations, dg::DG, cache)
    surface_flux = dg.surface_integral.surface_flux

    n_boundaries_per_direction = cache.boundaries.n_boundaries_per_direction
    neighbor_ids = cache.boundaries.neighbor_ids
    neighbor_sides = cache.boundaries.neighbor_sides
    orientations = cache.boundaries.orientations
    boundaries_u = cache.boundaries.u
    node_coordinates = cache.boundaries.node_coordinates
    surface_flux_values = cache.elements.surface_flux_values

    # Create new arrays on the GPU
    lasts = zero(n_boundaries_per_direction)
    firsts = zero(n_boundaries_per_direction)

    # May introduce kernel launching overhead
    last_first_indices_kernel = @cuda launch=false last_first_indices_kernel!(lasts, firsts,
                                                                              n_boundaries_per_direction)
    last_first_indices_kernel(lasts, firsts, n_boundaries_per_direction;
                              kernel_configurator_1d(last_first_indices_kernel, length(lasts))...)

    boundary_arr = CuArray{Int}(Array(firsts)[1]:Array(lasts)[end])
    indices_arr = firsts
    boundary_conditions_callable = replace_boundary_conditions(boundary_conditions)

    boundary_flux_kernel = @cuda launch=false boundary_flux_kernel!(surface_flux_values,
                                                                    boundaries_u, node_coordinates,
                                                                    t, boundary_arr, indices_arr,
                                                                    neighbor_ids, neighbor_sides,
                                                                    orientations,
                                                                    boundary_conditions_callable,
                                                                    equations, surface_flux)
    boundary_flux_kernel(surface_flux_values, boundaries_u, node_coordinates, t, boundary_arr,
                         indices_arr, neighbor_ids, neighbor_sides, orientations,
                         boundary_conditions_callable, equations, surface_flux;
                         kernel_configurator_2d(boundary_flux_kernel,
                                                size(surface_flux_values, 2)^2,
                                                length(boundary_arr))...)

    return nothing
end

# Dummy function for asserting mortars 
function cuda_prolong2mortars!(u, mesh::TreeMesh{3}, cache_mortars::False, dg::DG, cache)
    @assert iszero(length(cache.mortars.orientations))
end

# Pack kernels for prolonging solution to mortars
function cuda_prolong2mortars!(u, mesh::TreeMesh{3}, cache_mortars::True, dg::DG, cache)
    neighbor_ids = cache.mortars.neighbor_ids
    large_sides = cache.mortars.large_sides
    orientations = cache.mortars.orientations

    # The original CPU arrays hold NaNs
    u_upper_left = cache.mortars.u_upper_left
    u_upper_right = cache.mortars.u_upper_right
    u_lower_left = cache.mortars.u_lower_left
    u_lower_right = cache.mortars.u_lower_right
    forward_upper = dg.mortar.forward_upper
    forward_lower = dg.mortar.forward_lower

    prolong_mortars_small2small_kernel = @cuda launch=false prolong_mortars_small2small_kernel!(u_upper_left,
                                                                                                u_upper_right,
                                                                                                u_lower_left,
                                                                                                u_lower_right,
                                                                                                u,
                                                                                                neighbor_ids,
                                                                                                large_sides,
                                                                                                orientations)
    prolong_mortars_small2small_kernel(u_upper_left, u_upper_right, u_lower_left, u_lower_right, u,
                                       neighbor_ids, large_sides, orientations;
                                       kernel_configurator_3d(prolong_mortars_small2small_kernel,
                                                              size(u_upper_left, 2),
                                                              size(u_upper_left, 3)^2,
                                                              size(u_upper_left, 5))...)

    tmp_upper_left = zero(similar(u_upper_left)) # undef to zero
    tmp_upper_right = zero(similar(u_upper_right)) # undef to zero
    tmp_lower_left = zero(similar(u_lower_left)) # undef to zero
    tmp_lower_right = zero(similar(u_lower_right)) # undef to zero

    prolong_mortars_large2small_kernel = @cuda launch=false prolong_mortars_large2small_kernel!(tmp_upper_left,
                                                                                                tmp_upper_right,
                                                                                                tmp_lower_left,
                                                                                                tmp_lower_right,
                                                                                                u,
                                                                                                forward_upper,
                                                                                                forward_lower,
                                                                                                neighbor_ids,
                                                                                                large_sides,
                                                                                                orientations)
    prolong_mortars_large2small_kernel(tmp_upper_left, tmp_upper_right, tmp_lower_left,
                                       tmp_lower_right, u, forward_upper, forward_lower,
                                       neighbor_ids, large_sides, orientations;
                                       kernel_configurator_3d(prolong_mortars_large2small_kernel,
                                                              size(u_upper_left, 2),
                                                              size(u_upper_left, 3)^2,
                                                              size(u_upper_left, 5))...)

    prolong_mortars_large2small_kernel = @cuda launch=false prolong_mortars_large2small_kernel!(u_upper_left,
                                                                                                u_upper_right,
                                                                                                u_lower_left,
                                                                                                u_lower_right,
                                                                                                tmp_upper_left,
                                                                                                tmp_upper_right,
                                                                                                tmp_lower_left,
                                                                                                tmp_lower_right,
                                                                                                forward_upper,
                                                                                                forward_lower,
                                                                                                large_sides)
    prolong_mortars_large2small_kernel(u_upper_left, u_upper_right, u_lower_left, u_lower_right,
                                       tmp_upper_left, tmp_upper_right, tmp_lower_left,
                                       tmp_lower_right, forward_upper, forward_lower, large_sides;
                                       kernel_configurator_3d(prolong_mortars_large2small_kernel,
                                                              size(u_upper_left, 2),
                                                              size(u_upper_left, 3)^2,
                                                              size(u_upper_left, 5))...)

    return nothing
end

# Dummy function for asserting mortar fluxes
function cuda_mortar_flux!(mesh::TreeMesh{3}, cache_mortars::False, nonconservative_terms,
                           equations, dg::DG, cache)
    @assert iszero(length(cache.mortars.orientations))
end

# Pack kernels for calculating mortar fluxes
function cuda_mortar_flux!(mesh::TreeMesh{3}, cache_mortars::True, nonconservative_terms::False,
                           equations, dg::DG, cache)
    surface_flux = dg.surface_integral.surface_flux

    neighbor_ids = cache.mortars.neighbor_ids
    large_sides = cache.mortars.large_sides
    orientations = cache.mortars.orientations

    # The original CPU arrays hold NaNs
    u_upper_left = cache.mortars.u_upper_left
    u_upper_right = cache.mortars.u_upper_right
    u_lower_left = cache.mortars.u_lower_left
    u_lower_right = cache.mortars.u_lower_right
    reverse_upper = dg.mortar.reverse_upper
    reverse_lower = dg.mortar.reverse_lower

    surface_flux_values = cache.elements.surface_flux_values
    tmp_surface_flux_values = zero(similar(surface_flux_values)) # undef to zero

    fstar_primary_upper_left = cache.fstar_primary_upper_left
    fstar_primary_upper_right = cache.fstar_primary_upper_right
    fstar_primary_lower_left = cache.fstar_primary_lower_left
    fstar_primary_lower_right = cache.fstar_primary_lower_right
    fstar_secondary_upper_left = cache.fstar_secondary_upper_left
    fstar_secondary_upper_right = cache.fstar_secondary_upper_right
    fstar_secondary_lower_left = cache.fstar_secondary_lower_left
    fstar_secondary_lower_right = cache.fstar_secondary_lower_right

    mortar_flux_kernel = @cuda launch=false mortar_flux_kernel!(fstar_primary_upper_left,
                                                                fstar_primary_upper_right,
                                                                fstar_primary_lower_left,
                                                                fstar_primary_lower_right,
                                                                fstar_secondary_upper_left,
                                                                fstar_secondary_upper_right,
                                                                fstar_secondary_lower_left,
                                                                fstar_secondary_lower_right,
                                                                u_upper_left, u_upper_right,
                                                                u_lower_left, u_lower_right,
                                                                orientations, equations,
                                                                surface_flux)
    mortar_flux_kernel(fstar_primary_upper_left, fstar_primary_upper_right,
                       fstar_primary_lower_left, fstar_primary_lower_right,
                       fstar_secondary_upper_left, fstar_secondary_upper_right,
                       fstar_secondary_lower_left, fstar_secondary_lower_right,
                       u_upper_left, u_upper_right, u_lower_left, u_lower_right, orientations,
                       equations, surface_flux;
                       kernel_configurator_3d(mortar_flux_kernel, size(u_upper_left, 3),
                                              size(u_upper_left, 4),
                                              length(orientations))...)

    tmp_upper_left = zero(similar(surface_flux_values)) # undef to zero
    tmp_upper_right = zero(similar(surface_flux_values)) # undef to zero
    tmp_lower_left = zero(similar(surface_flux_values)) # undef to zero
    tmp_lower_right = zero(similar(surface_flux_values)) # undef to zero

    mortar_flux_copy_to_kernel = @cuda launch=false mortar_flux_copy_to_kernel!(surface_flux_values,
                                                                                tmp_upper_left,
                                                                                tmp_upper_right,
                                                                                tmp_lower_left,
                                                                                tmp_lower_right,
                                                                                fstar_primary_upper_left,
                                                                                fstar_primary_upper_right,
                                                                                fstar_primary_lower_left,
                                                                                fstar_primary_lower_right,
                                                                                fstar_secondary_upper_left,
                                                                                fstar_secondary_upper_right,
                                                                                fstar_secondary_lower_left,
                                                                                fstar_secondary_lower_right,
                                                                                reverse_upper,
                                                                                reverse_lower,
                                                                                neighbor_ids,
                                                                                large_sides,
                                                                                orientations)
    mortar_flux_copy_to_kernel(surface_flux_values, tmp_upper_left, tmp_upper_right, tmp_lower_left,
                               tmp_lower_right, fstar_primary_upper_left, fstar_primary_upper_right,
                               fstar_primary_lower_left, fstar_primary_lower_right,
                               fstar_secondary_upper_left, fstar_secondary_upper_right,
                               fstar_secondary_lower_left, fstar_secondary_lower_right,
                               reverse_upper, reverse_lower, neighbor_ids, large_sides,
                               orientations;
                               kernel_configurator_3d(mortar_flux_copy_to_kernel,
                                                      size(surface_flux_values, 1),
                                                      size(surface_flux_values, 2)^2,
                                                      length(orientations))...)

    mortar_flux_copy_to_kernel = @cuda launch=false mortar_flux_copy_to_kernel!(surface_flux_values,
                                                                                tmp_surface_flux_values,
                                                                                tmp_upper_left,
                                                                                tmp_upper_right,
                                                                                tmp_lower_left,
                                                                                tmp_lower_right,
                                                                                reverse_upper,
                                                                                reverse_lower,
                                                                                neighbor_ids,
                                                                                large_sides,
                                                                                orientations,
                                                                                equations)
    mortar_flux_copy_to_kernel(surface_flux_values, tmp_surface_flux_values, tmp_upper_left,
                               tmp_upper_right, tmp_lower_left, tmp_lower_right, reverse_upper,
                               reverse_lower, neighbor_ids, large_sides, orientations, equations;
                               kernel_configurator_3d(mortar_flux_copy_to_kernel,
                                                      size(surface_flux_values, 1),
                                                      size(surface_flux_values, 2)^2,
                                                      length(orientations))...)

    return nothing
end

# Pack kernels for calculating mortar fluxes
function cuda_mortar_flux!(mesh::TreeMesh{3}, cache_mortars::True, nonconservative_terms::True,
                           equations, dg::DG, cache)
    surface_flux, nonconservative_flux = dg.surface_integral.surface_flux

    neighbor_ids = cache.mortars.neighbor_ids
    large_sides = cache.mortars.large_sides
    orientations = cache.mortars.orientations

    # The original CPU arrays hold NaNs
    u_upper_left = cache.mortars.u_upper_left
    u_upper_right = cache.mortars.u_upper_right
    u_lower_left = cache.mortars.u_lower_left
    u_lower_right = cache.mortars.u_lower_right
    reverse_upper = dg.mortar.reverse_upper
    reverse_lower = dg.mortar.reverse_lower

    surface_flux_values = cache.elements.surface_flux_values
    tmp_surface_flux_values = zero(similar(surface_flux_values)) # undef to zero

    fstar_primary_upper_left = cache.fstar_primary_upper_left
    fstar_primary_upper_right = cache.fstar_primary_upper_right
    fstar_primary_lower_left = cache.fstar_primary_lower_left
    fstar_primary_lower_right = cache.fstar_primary_lower_right
    fstar_secondary_upper_left = cache.fstar_secondary_upper_left
    fstar_secondary_upper_right = cache.fstar_secondary_upper_right
    fstar_secondary_lower_left = cache.fstar_secondary_lower_left
    fstar_secondary_lower_right = cache.fstar_secondary_lower_right

    mortar_flux_kernel = @cuda launch=false mortar_flux_kernel!(fstar_primary_upper_left,
                                                                fstar_primary_upper_right,
                                                                fstar_primary_lower_left,
                                                                fstar_primary_lower_right,
                                                                fstar_secondary_upper_left,
                                                                fstar_secondary_upper_right,
                                                                fstar_secondary_lower_left,
                                                                fstar_secondary_lower_right,
                                                                u_upper_left, u_upper_right,
                                                                u_lower_left, u_lower_right,
                                                                orientations, large_sides,
                                                                equations, surface_flux,
                                                                nonconservative_flux)
    mortar_flux_kernel(fstar_primary_upper_left, fstar_primary_upper_right,
                       fstar_primary_lower_left, fstar_primary_lower_right,
                       fstar_secondary_upper_left, fstar_secondary_upper_right,
                       fstar_secondary_lower_left, fstar_secondary_lower_right,
                       u_upper_left, u_upper_right, u_lower_left, u_lower_right, orientations,
                       large_sides, equations, surface_flux, nonconservative_flux;
                       kernel_configurator_3d(mortar_flux_kernel, size(u_upper_left, 3),
                                              size(u_upper_left, 4),
                                              length(orientations))...)

    tmp_upper_left = zero(similar(surface_flux_values)) # undef to zero
    tmp_upper_right = zero(similar(surface_flux_values)) # undef to zero
    tmp_lower_left = zero(similar(surface_flux_values)) # undef to zero
    tmp_lower_right = zero(similar(surface_flux_values)) # undef to zero

    mortar_flux_copy_to_kernel = @cuda launch=false mortar_flux_copy_to_kernel!(surface_flux_values,
                                                                                tmp_upper_left,
                                                                                tmp_upper_right,
                                                                                tmp_lower_left,
                                                                                tmp_lower_right,
                                                                                fstar_primary_upper_left,
                                                                                fstar_primary_upper_right,
                                                                                fstar_primary_lower_left,
                                                                                fstar_primary_lower_right,
                                                                                fstar_secondary_upper_left,
                                                                                fstar_secondary_upper_right,
                                                                                fstar_secondary_lower_left,
                                                                                fstar_secondary_lower_right,
                                                                                reverse_upper,
                                                                                reverse_lower,
                                                                                neighbor_ids,
                                                                                large_sides,
                                                                                orientations)
    mortar_flux_copy_to_kernel(surface_flux_values, tmp_upper_left, tmp_upper_right, tmp_lower_left,
                               tmp_lower_right, fstar_primary_upper_left, fstar_primary_upper_right,
                               fstar_primary_lower_left, fstar_primary_lower_right,
                               fstar_secondary_upper_left, fstar_secondary_upper_right,
                               fstar_secondary_lower_left, fstar_secondary_lower_right,
                               reverse_upper, reverse_lower, neighbor_ids, large_sides,
                               orientations;
                               kernel_configurator_3d(mortar_flux_copy_to_kernel,
                                                      size(surface_flux_values, 1),
                                                      size(surface_flux_values, 2)^2,
                                                      length(orientations))...)

    mortar_flux_copy_to_kernel = @cuda launch=false mortar_flux_copy_to_kernel!(surface_flux_values,
                                                                                tmp_surface_flux_values,
                                                                                tmp_upper_left,
                                                                                tmp_upper_right,
                                                                                tmp_lower_left,
                                                                                tmp_lower_right,
                                                                                reverse_upper,
                                                                                reverse_lower,
                                                                                neighbor_ids,
                                                                                large_sides,
                                                                                orientations,
                                                                                equations)
    mortar_flux_copy_to_kernel(surface_flux_values, tmp_surface_flux_values, tmp_upper_left,
                               tmp_upper_right, tmp_lower_left, tmp_lower_right, reverse_upper,
                               reverse_lower, neighbor_ids, large_sides, orientations, equations;
                               kernel_configurator_3d(mortar_flux_copy_to_kernel,
                                                      size(surface_flux_values, 1),
                                                      size(surface_flux_values, 2)^2,
                                                      length(orientations))...)

    return nothing
end

# Pack kernels for calculating surface integrals
function cuda_surface_integral!(du, mesh::TreeMesh{3}, equations, dg::DG, cache)
    factor_arr = CuArray([
                             dg.basis.boundary_interpolation[1, 1],
                             dg.basis.boundary_interpolation[size(du, 2), 2]
                         ])
    surface_flux_values = cache.elements.surface_flux_values

    surface_integral_kernel = @cuda launch=false surface_integral_kernel!(du, factor_arr,
                                                                          surface_flux_values,
                                                                          equations)
    surface_integral_kernel(du, factor_arr, surface_flux_values, equations;
                            kernel_configurator_3d(surface_integral_kernel, size(du, 1),
                                                   size(du, 2)^3, size(du, 5))...)

    return nothing
end

# Pack kernels for applying Jacobian to reference element
function cuda_jacobian!(du, mesh::TreeMesh{3}, equations, cache)
    inverse_jacobian = cache.elements.inverse_jacobian

    jacobian_kernel = @cuda launch=false jacobian_kernel!(du, inverse_jacobian, equations)
    jacobian_kernel(du, inverse_jacobian, equations;
                    kernel_configurator_3d(jacobian_kernel, size(du, 1), size(du, 2)^3,
                                           size(du, 5))...)

    return nothing
end

# Dummy function returning nothing            
function cuda_sources!(du, u, t, source_terms::Nothing, equations::AbstractEquations{3}, cache)
    return nothing
end

# Pack kernels for calculating source terms 
function cuda_sources!(du, u, t, source_terms, equations::AbstractEquations{3}, cache)
    node_coordinates = cache.elements.node_coordinates

    source_terms_kernel = @cuda launch=false source_terms_kernel!(du, u, node_coordinates, t,
                                                                  equations, source_terms)
    source_terms_kernel(du, u, node_coordinates, t, equations, source_terms;
                        kernel_configurator_2d(source_terms_kernel, size(u, 2)^3, size(u, 5))...)

    return nothing
end

# Put everything together into a single function.

# See also `rhs!` function in Trixi.jl
function rhs_gpu!(du, u, t, mesh::TreeMesh{3}, equations, boundary_conditions,
                  source_terms::Source, dg::DG, cache_gpu, cache_cpu) where {Source}
    # reset_du!(du) 
    # reset_du!(du) is now fused into the next kernel, 
    # removing the need for a separate function call.

    cuda_volume_integral!(du, u, mesh, have_nonconservative_terms(equations), equations,
                          dg.volume_integral, dg, cache_gpu, cache_cpu)

    cuda_prolong2interfaces!(u, mesh, equations, cache_gpu)

    cuda_interface_flux!(mesh, have_nonconservative_terms(equations), equations, dg, cache_gpu)

    cuda_prolong2boundaries!(u, mesh, boundary_conditions, equations, cache_gpu)

    cuda_boundary_flux!(t, mesh, boundary_conditions,
                        have_nonconservative_terms(equations), equations, dg, cache_gpu)

    cuda_prolong2mortars!(u, mesh, check_cache_mortars(cache_gpu), dg, cache_gpu)

    cuda_mortar_flux!(mesh, check_cache_mortars(cache_gpu), have_nonconservative_terms(equations),
                      equations, dg, cache_gpu)

    cuda_surface_integral!(du, mesh, equations, dg, cache_gpu)

    cuda_jacobian!(du, mesh, equations, cache_gpu)

    cuda_sources!(du, u, t, source_terms, equations, cache_gpu)

    return nothing
end
