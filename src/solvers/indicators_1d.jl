# Currently hold cache for indicators on CPU
# Note that this should be optimized on GPU in the future.
function create_cache(::Type{IndicatorHennemannGassner},
                      equations::AbstractEquations{1}, basis::LobattoLegendreBasisGPU)
    alpha = Vector{real(basis)}()
    alpha_tmp = similar(alpha)

    A = Array{real(basis), ndims(equations)}

    # Single indicator and modal arrays (single thread)
    # Note: This can be optimized by moving to GPU but need to be careful about
    # its actual performance.
    indicator_threaded = [A(undef, nnodes(basis))]
    modal_threaded = [A(undef, nnodes(basis))]

    return (; alpha, alpha_tmp, indicator_threaded, modal_threaded)
end
