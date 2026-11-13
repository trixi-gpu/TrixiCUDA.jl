# Keep the fused kernels for classic HG. Generalized HG selects the operator
# per element, following Trixi's dgsem/calc_volume_integral.jl.
function classic_hg(integral)
    integral.volume_integral_default === integral.volume_integral_blend_high_order &&
        integral.volume_integral_blend_high_order isa VolumeIntegralFluxDifferencing &&
        integral.volume_integral_blend_low_order isa VolumeIntegralPureLGLFiniteVolume
end

# Trixi's reconstruction functions need only nodes and nnodes from the solver.
# This small isbits basis lets us call them on device without a host DG cache.
struct HGReconstructionBasis{RealT, N} <: AbstractBasisSBP{RealT}
    nodes::SVector{N, RealT}
end
nnodes(::HGReconstructionBasis{RealT, N}) where {RealT, N} = N

function calc_volume_integral_generalized_hg!(du, u, mesh::TreeMesh{D}, nonconservative_terms,
                                              equations, integral, dg, cache_cpu) where {D}
    default = integral.volume_integral_default
    high = integral.volume_integral_blend_high_order
    low = integral.volume_integral_blend_low_order
    for operator in (default, high)
        operator isa Union{VolumeIntegralWeakForm, VolumeIntegralFluxDifferencing} ||
            throw(ArgumentError("GPU HG requires weak-form or flux-differencing high-order operators"))
        nonconservative_terms isa True && operator isa VolumeIntegralWeakForm &&
            throw(ArgumentError("Weak-form HG operators require conservative equations"))
    end
    low isa Union{VolumeIntegralPureLGLFiniteVolume, VolumeIntegralPureLGLFiniteVolumeO2} ||
        throw(ArgumentError("GPU HG requires first- or second-order subcell FV"))
    nonconservative_terms isa True && low isa VolumeIntegralPureLGLFiniteVolumeO2 &&
        throw(ArgumentError("Second-order subcell FV requires conservative equations, as in Trixi"))

    alpha_host = integral.indicator(Array(u), mesh, equations, dg, cache_cpu)
    RealT = eltype(alpha_host)
    atol = max(100 * eps(RealT), eps(RealT)^convert(RealT, 0.75f0))
    alpha = CuArray(alpha_host)
    basis = HGReconstructionBasis(SVector{nnodes(dg)}(dg.basis.nodes))
    reconstruction_dg = DGSEM{typeof(basis), Nothing, Nothing, Nothing}(basis, nothing, nothing, nothing)
    count = length(u) ÷ nvariables(equations)
    threads = min(128, MAX_THREADS_PER_BLOCK)
    @cuda threads=threads blocks=cld(count, threads) generalized_hg_kernel!(du, u, alpha, atol,
                                                                            equations, nonconservative_terms,
                                                                            default, high, low, reconstruction_dg,
                                                                            dg.basis.derivative_hat,
                                                                            dg.basis.derivative_split,
                                                                            dg.basis.inverse_weights, Val(D))
    return nothing
end

function generalized_hg_kernel!(du, u, alpha, atol, equations, nonconservative_terms,
                                default, high, low, dg, derivative_hat, derivative_split,
                                inverse_weights, ::Val{D}) where {D}
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n = nnodes(dg)
    if index <= length(u) ÷ nvariables(equations)
        element = (index - 1) ÷ n^D + 1
        node = ntuple(d -> ((index - 1) ÷ n^(d - 1)) % n + 1, Val(D))
        a = alpha[element]
        if isapprox(a, 0, atol = atol)
            value = hg_node_volume(u, node, element, equations, nonconservative_terms,
                                   default, dg, derivative_hat, derivative_split)
        else
            value = (1 - a) * hg_node_volume(u, node, element, equations, nonconservative_terms,
                                             high, dg, derivative_hat, derivative_split)
            value += a * hg_node_fv(u, node, element, equations, nonconservative_terms,
                                    low, dg, inverse_weights)
        end
        for v in 1:nvariables(equations)
            @inbounds du[v, node..., element] = value[v]
        end
    end
    return nothing
end

@inline function hg_node_volume(u, node::NTuple{D}, element, equations, ::False,
                                ::VolumeIntegralWeakForm, dg, derivative_hat, derivative_split) where {D}
    value = zero(get_node_vars(u, equations, node..., element))
    for direction in 1:D, i in 1:nnodes(dg)
        other = Base.setindex(node, i, direction)
        state = get_node_vars(u, equations, other..., element)
        value += derivative_hat[node[direction], i] * flux(state, direction, equations)
    end
    return value
end

@inline hg_split_flux(left, right, direction, equations, ::False, volume_flux) = volume_flux(left, right, direction, equations)
@inline function hg_split_flux(left, right, direction, equations, ::True, volume_flux)
    symmetric, nonconservative = volume_flux
    return symmetric(left, right, direction, equations) +
           0.5f0 * nonconservative(left, right, direction, equations)
end

@inline function hg_node_volume(u, node::NTuple{D}, element, equations, nonconservative_terms,
                                integral::VolumeIntegralFluxDifferencing, dg,
                                derivative_hat, derivative_split) where {D}
    state = get_node_vars(u, equations, node..., element)
    value = zero(state)
    for direction in 1:D, i in 1:nnodes(dg)
        if i != node[direction]
            other = Base.setindex(node, i, direction)
            neighbor = get_node_vars(u, equations, other..., element)
            value += derivative_split[node[direction], i] *
                     hg_split_flux(state, neighbor, direction, equations,
                                   nonconservative_terms, integral.volume_flux)
        end
    end
    return value
end

# Left/right states at an internal subcell interface. The outer interfaces have
# zero volume flux; the existing surface integral supplies the element flux.
@inline function hg_reconstruct(u, node, element, direction, i, equations,
                                ::VolumeIntegralPureLGLFiniteVolume, dg)
    left = get_node_vars(u, equations, Base.setindex(node, i - 1, direction)..., element)
    right = get_node_vars(u, equations, Base.setindex(node, i, direction)..., element)
    return left, right
end

@inline function hg_reconstruct(u, node, element, direction, i, equations,
                                integral::VolumeIntegralPureLGLFiniteVolumeO2, dg)
    state(j) = integral.cons2recon(get_node_vars(u, equations,
                                                 Base.setindex(node, j, direction)..., element), equations)
    left, right = integral.reconstruction_mode(state(max(1, i - 2)), state(i - 1),
                                               state(i), state(min(nnodes(dg), i + 1)),
                                               integral.sc_interface_coords, i,
                                               integral.slope_limiter, dg)
    return integral.recon2cons(left, equations), integral.recon2cons(right, equations)
end

@inline function hg_node_fv(u, node::NTuple{D}, element, equations, nonconservative_terms,
                            integral, dg, inverse_weights) where {D}
    state = get_node_vars(u, equations, node..., element)
    value = zero(state)
    for direction in 1:D
        i = node[direction]
        if i < nnodes(dg)
            left, right = hg_reconstruct(u, node, element, direction, i + 1, equations, integral, dg)
            value += inverse_weights[i] * hg_split_flux(left, right, direction, equations,
                                                        nonconservative_terms, integral.volume_flux_fv)
        end
        if i > 1
            left, right = hg_reconstruct(u, node, element, direction, i, equations, integral, dg)
            value -= inverse_weights[i] * hg_right_flux(left, right, direction, equations,
                                                        nonconservative_terms, integral.volume_flux_fv)
        end
    end
    return value
end

@inline hg_right_flux(left, right, direction, equations, ::False, volume_flux) = volume_flux(left, right, direction, equations)
@inline function hg_right_flux(left, right, direction, equations, ::True, volume_flux)
    symmetric, nonconservative = volume_flux
    return symmetric(left, right, direction, equations) +
           0.5f0 * nonconservative(right, left, direction, equations)
end
