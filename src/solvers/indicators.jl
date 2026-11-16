# Part of the indicator functions for degraded `DG` type.
# Note that maybe we should consider optimizing these original CPU code to
# get it better run on GPU.

# Compute the raw indicator values on the host. The smoothing step from Trixi.jl is
# applied on the GPU afterwards, see `calc_indicator_gpu!` and `apply_smoothing_gpu!`.
function (indicator_hg::IndicatorHennemannGassner)(u, mesh, equations, dg::DGSEM{<:LobattoLegendreBasisGPU}, cache;
                                                   kwargs...)
    (; alpha) = indicator_hg.cache
    # TODO: Taal refactor, when to `resize!` stuff changed possibly by AMR?
    #       Shall we implement `resize!(semi::AbstractSemidiscretization, new_size)`
    #       or just `resize!` whenever we call the relevant methods as we do now?
    resize!(alpha, nelements(dg, cache))

    # magic parameters
    # TODO: Are there better values for Float32?
    RealT = real(dg)
    threshold = 0.5f0 * 10^(convert(RealT, -1.8) * nnodes(dg)^convert(RealT, 0.25))
    o_0001 = convert(RealT, 0.0001)
    parameter_s = log((1 - o_0001) / o_0001)

    # Degrade to single thread here (origin calls `@threaded` in Trixi.jl)
    for element in eachelement(dg, cache)
        # This is dispatched by mesh dimension.
        # Use this function barrier and unpack inside to avoid passing closures to
        # Polyester.jl with `@batch` (`@threaded`).
        # Otherwise, `@threaded` does not work here with Julia ARM on macOS.
        # See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
        calc_indicator_hennemann_gassner!(indicator_hg, threshold, parameter_s, u,
                                          element, mesh, equations, dg, cache)
    end

    return alpha
end

function calc_indicator_gpu!(indicator_hg::IndicatorHennemannGassner, u, mesh, equations,
                             dg::DGSEM, cache)
    (; alpha, alpha_tmp) = cache
    # TODO: The indicator values are computed on the host, so `u` is copied to the host
    # (with a new allocation) on every RHS call. Compute them on the GPU instead, or at 
    # least reuse a preallocated host buffer.
    copyto!(alpha, indicator_hg(Array(u), mesh, equations, dg, cache))

    if indicator_hg.alpha_smooth
        apply_smoothing_gpu!(mesh, alpha, alpha_tmp, dg, cache)
    end

    return alpha
end

function apply_smoothing_gpu!(mesh::TreeMesh{1}, alpha, alpha_tmp, dg::DGSEM, cache)
    # Copy alpha values such that smoothing is independent of the element access order
    copyto!(alpha_tmp, alpha)

    smooth_interfaces_gpu!(alpha, alpha_tmp, cache.interfaces.neighbor_ids)

    return nothing
end

function apply_smoothing_gpu!(mesh::Union{TreeMesh{2}, TreeMesh{3}}, alpha, alpha_tmp, dg::DGSEM, cache)
    # Copy alpha values such that smoothing is independent of the element access order
    copyto!(alpha_tmp, alpha)

    smooth_interfaces_gpu!(alpha, alpha_tmp, cache.interfaces.neighbor_ids)
    smooth_mortars_gpu!(alpha, alpha_tmp, cache.mortars.neighbor_ids)

    return nothing
end

function smooth_interfaces_gpu!(alpha, alpha_tmp, neighbor_ids)
    isempty(neighbor_ids) && return nothing

    smoothing_interfaces_kernel = @cuda launch=false smoothing_interfaces_kernel!(alpha, alpha_tmp, neighbor_ids)
    smoothing_interfaces_kernel(alpha, alpha_tmp, neighbor_ids;
                                kernel_configurator_1d(smoothing_interfaces_kernel, size(neighbor_ids, 2))...)

    return nothing
end

function smooth_mortars_gpu!(alpha, alpha_tmp, neighbor_ids)
    isempty(neighbor_ids) && return nothing

    smoothing_mortars_kernel = @cuda launch=false smoothing_mortars_kernel!(alpha, alpha_tmp, neighbor_ids)
    smoothing_mortars_kernel(alpha, alpha_tmp, neighbor_ids;
                             kernel_configurator_1d(smoothing_mortars_kernel, size(neighbor_ids, 2))...)

    return nothing
end

# Kernels for the smoothing. Each thread handles one interface or one mortar and raises
# the alpha values of its neighbors with an atomic `max`. All values on the right-hand side
# are read from `alpha_tmp`, which is not modified here, so the result does not depend on
# the order in which the threads run and matches the serial loop in Trixi.jl.

# Kernel for smoothing across interfaces
function smoothing_interfaces_kernel!(alpha, alpha_tmp, neighbor_ids)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if k <= size(neighbor_ids, 2)
        @inbounds left = neighbor_ids[1, k]
        @inbounds right = neighbor_ids[2, k]

        @inbounds CUDA.@atomic alpha[left] = max(alpha[left], 0.5f0 * alpha_tmp[right])
        @inbounds CUDA.@atomic alpha[right] = max(alpha[right], 0.5f0 * alpha_tmp[left])
    end

    return nothing
end

# Kernel for smoothing across L2 mortars. The last row of `neighbor_ids` holds the large
# element and the rows before it hold the small elements (2 in 2D, 4 in 3D).
function smoothing_mortars_kernel!(alpha, alpha_tmp, neighbor_ids)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if k <= size(neighbor_ids, 2)
        n_small = size(neighbor_ids, 1) - 1
        @inbounds large = neighbor_ids[n_small + 1, k]

        for s in 1:n_small
            @inbounds small = neighbor_ids[s, k]

            @inbounds CUDA.@atomic alpha[small] = max(alpha[small], 0.5f0 * alpha_tmp[large])
            @inbounds CUDA.@atomic alpha[large] = max(alpha[large], 0.5f0 * alpha_tmp[small])
        end
    end

    return nothing
end
