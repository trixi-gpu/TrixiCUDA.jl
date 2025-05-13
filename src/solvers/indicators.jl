# Part of the indicator functions for degraded `DG` type.
# Note that maybe we should consider optimizing these original CPU code to
# get it better run on GPU.

# Same as `IndicatorHennemannGassner` in Trixi.jl but with `DG` signature
# How to implement this in a better way?
function (indicator_hg::IndicatorHennemannGassner)(u, mesh, equations, dg::DG, cache;
                                                   kwargs...)
    (; alpha_smooth) = indicator_hg
    (; alpha, alpha_tmp) = indicator_hg.cache
    # TODO: Taal refactor, when to `resize!` stuff changed possibly by AMR?
    #       Shall we implement `resize!(semi::AbstractSemidiscretization, new_size)`
    #       or just `resize!` whenever we call the relevant methods as we do now?
    resize!(alpha, nelements(dg, cache))
    if alpha_smooth
        resize!(alpha_tmp, nelements(dg, cache))
    end

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

    if alpha_smooth
        apply_smoothing!(mesh, alpha, alpha_tmp, dg, cache)
    end

    return alpha
end
