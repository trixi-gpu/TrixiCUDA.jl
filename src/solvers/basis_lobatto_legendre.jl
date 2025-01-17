# Everything related to Lobatto-Legendre basis adapted for initialization on the GPU.

# Similar to `LobattoLegendreBasis` in Trixi.jl but GPU compatible
# struct LobattoLegendreBasisGPU{RealT <: Real, NNODES,
#                             VectorT <: AbstractGPUVector{RealT},
#                             InverseVandermondeLegendre <: AbstractGPUMatrix{RealT},
#                             BoundaryMatrix <: AbstractGPUMatrix{RealT},
#                                DerivativeMatrix <: AbstractGPUMatrix{RealT}} <: AbstractBasisSBP{RealT}
#     nodes::VectorT
#     weights::VectorT
#     inverse_weights::VectorT

#     inverse_vandermonde_legendre::InverseVandermondeLegendre
#     boundary_interpolation::BoundaryMatrix # lhat

#     derivative_matrix::DerivativeMatrix # strong form derivative matrix
#     derivative_split::DerivativeMatrix # strong form derivative matrix minus boundary terms
#     derivative_split_transpose::DerivativeMatrix # transpose of `derivative_split`
#     derivative_dhat::DerivativeMatrix # weak form matrix dhat
#     # negative adjoint wrt the SBP dot product
# end

# Similar to `LobattoLegendreBasis` in Trixi.jl 
function LobattoLegendreBasisGPU(RealT, polydeg::Integer)
    nnodes_ = polydeg + 1

    # TODO: Use GPU kernels to complete the computation (compare with CPU)
    nodes_, weights_ = gauss_lobatto_nodes_weights(nnodes_, RealT)
    inverse_weights_ = inv.(weights_)

    _, inverse_vandermonde_legendre_ = vandermonde_legendre(nodes_, RealT)

    boundary_interpolation_ = zeros(RealT, nnodes_, 2)
    boundary_interpolation_[:, 1] = calc_lhat(-one(RealT), nodes_, weights_)
    boundary_interpolation_[:, 2] = calc_lhat(one(RealT), nodes_, weights_)

    derivative_matrix_ = polynomial_derivative_matrix(nodes_)
    derivative_split_ = calc_dsplit(nodes_, weights_)
    derivative_dhat_ = calc_dhat(nodes_, weights_)

    # Convert to GPU arrays
    # nodes = CuArray(nodes_) # to avoid scalar indexing caused by `init_elements`
    weights = CuArray(weights_)
    inverse_weights = CuArray(inverse_weights_)

    inverse_vandermonde_legendre = CuArray(inverse_vandermonde_legendre_)
    boundary_interpolation = CuArray(boundary_interpolation_)

    derivative_matrix = CuArray(derivative_matrix_)
    derivative_split = CuArray(derivative_split_)
    derivative_split_transpose = CuArray(derivative_split_')
    derivative_dhat = CuArray(derivative_dhat_)

    # TODO: Return GPU type
    return LobattoLegendreBasis{RealT, nnodes_, typeof(nodes_),
                                typeof(inverse_vandermonde_legendre),
                                typeof(boundary_interpolation),
                                typeof(derivative_matrix)}(nodes_, weights,
                                                           inverse_weights,
                                                           inverse_vandermonde_legendre,
                                                           boundary_interpolation,
                                                           derivative_matrix,
                                                           derivative_split,
                                                           derivative_split_transpose,
                                                           derivative_dhat)
end

# @inline Base.real(basis::LobattoLegendreBasisGPU{RealT}) where {RealT} = RealT

# @inline function nnodes(basis::LobattoLegendreBasisGPU{RealT, NNODES}) where {RealT, NNODES}
#     NNODES
# end

# @inline get_nodes(basis::LobattoLegendreBasisGPU) = basis.nodes

# # Similar to `LobattoLegendreMortarL2` in Trixi.jl but GPU compatible
# struct LobattoLegendreMortarL2GPU{RealT <: Real, NNODES,
#                                ForwardMatrix <: AbstractGPUMatrix{RealT},
#                                ReverseMatrix <: AbstractGPUMatrix{RealT}} <: AbstractMortarL2{RealT}
#     forward_upper::ForwardMatrix
#     forward_lower::ForwardMatrix
#     reverse_upper::ReverseMatrix
#     reverse_lower::ReverseMatrix
# end

# Similar to `MortarL2` in Trixi.jl
function MortarL2GPU(basis::LobattoLegendreBasis)
    RealT = real(basis)
    nnodes_ = nnodes(basis)

    forward_upper_ = calc_forward_upper(nnodes_, RealT)
    forward_lower_ = calc_forward_lower(nnodes_, RealT)
    reverse_upper_ = calc_reverse_upper(nnodes_, Val(:gauss), RealT)
    reverse_lower_ = calc_reverse_lower(nnodes_, Val(:gauss), RealT)

    # Convert to GPU arrays
    # TODO: `RealT` can be removed once the upstream type instability issue is fixed,
    # and the Trixi.jl version can then be updated to the latest one.
    forward_upper = CuArray{RealT}(forward_upper_)
    forward_lower = CuArray{RealT}(forward_lower_)
    reverse_upper = CuArray{RealT}(reverse_upper_)
    reverse_lower = CuArray{RealT}(reverse_lower_)

    # TODO: Return GPU type
    LobattoLegendreMortarL2{RealT, nnodes_, typeof(forward_upper),
                            typeof(reverse_upper)}(forward_upper, forward_lower,
                                                   reverse_upper, reverse_lower)
end
