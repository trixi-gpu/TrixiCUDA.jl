# Everything related to Lobatto-Legendre basis adapted for initialization on the GPU.

# CPU version of LobattoLegendreBasis from Trixi.jl (for reference)
# struct LobattoLegendreBasis{RealT <: Real, NNODES,
#                             VectorT <: AbstractVector{RealT},
#                             InverseVandermondeLegendre <: AbstractMatrix{RealT},
#                             BoundaryMatrix <: AbstractMatrix{RealT},
#                             DerivativeMatrix <: AbstractMatrix{RealT}} <:
#        AbstractBasisSBP{RealT}
#     nodes::VectorT
#     weights::VectorT
#     inverse_weights::VectorT

#     inverse_vandermonde_legendre::InverseVandermondeLegendre
#     boundary_interpolation::BoundaryMatrix # lhat

#     derivative_matrix::DerivativeMatrix # strong form derivative matrix
#     derivative_split::DerivativeMatrix # strong form derivative matrix minus boundary terms
#     derivative_split_transpose::DerivativeMatrix # transpose of `derivative_split`
#     derivative_dhat::DerivativeMatrix # weak form matrix "dhat",
#     # negative adjoint wrt the SBP dot product
# end

# Similar to `LobattoLegendreBasis` in Trixi.jl 
struct LobattoLegendreBasisGPU{RealT <: Real, NNODES,
                               VectorT <: CuArray{RealT, 1},
                               InverseVandermondeLegendre <: AbstractMatrix{RealT},
                               BoundaryMatrix <: AbstractMatrix{RealT},
                               DerivativeMatrix <: CuArray{RealT, 2}} <: AbstractBasisSBP{RealT}
    # CPU array, no need to convert to GPU array
    # Will it be better if adapt a CPU type?
    nodes::Any
    weights::Any

    # Set as GPU array type
    inverse_weights::VectorT

    # CPU array, can adapt to a GPU type if needed
    inverse_vandermonde_legendre::InverseVandermondeLegendre
    boundary_interpolation::BoundaryMatrix

    # Set as GPU array type
    derivative_matrix::DerivativeMatrix
    derivative_split::DerivativeMatrix
    derivative_split_transpose::DerivativeMatrix
    derivative_dhat::DerivativeMatrix
end

"""
    LobattoLegendreBasisGPU([RealT = Float64,] polydeg::Integer)

Create a nodal Lobatto-Legendre basis for polynomials of degree `polydeg`, partially storing 
and operating on GPU arrays (`CuArray`). This basis is designed for GPU-accelerated discontinuous 
Galerkin spectral element method (DGSEM).

!!! warning "Experimental implementation"
    This is an experimental feature and may change or be removed in future releases due to 
    ongoing performance optimizations.
"""
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
    inverse_weights = CuArray{RealT}(inverse_weights_)

    derivative_matrix = CuArray{RealT}(derivative_matrix_)
    derivative_split = CuArray{RealT}(derivative_split_)
    derivative_split_transpose = CuArray{RealT}(derivative_split_')
    derivative_dhat = CuArray{RealT}(derivative_dhat_)

    # Avoid scalar indexing
    # inverse_vandermonde_legendre = CuArray{RealT}(inverse_vandermonde_legendre_)
    # boundary_interpolation = CuArray(boundary_interpolation_)

    return LobattoLegendreBasisGPU{RealT, nnodes_, typeof(inverse_weights),
                                   typeof(inverse_vandermonde_legendre_),
                                   typeof(boundary_interpolation_),
                                   typeof(derivative_matrix)}(nodes_, weights_,
                                                              inverse_weights,
                                                              inverse_vandermonde_legendre_,
                                                              boundary_interpolation_,
                                                              derivative_matrix,
                                                              derivative_split,
                                                              derivative_split_transpose,
                                                              derivative_dhat)
end

LobattoLegendreBasisGPU(polydeg::Integer) = LobattoLegendreBasisGPU(Float64, polydeg) # Float32 ?

@inline Base.real(basis::LobattoLegendreBasisGPU{RealT}) where {RealT} = RealT

@inline function nnodes(basis::LobattoLegendreBasisGPU{RealT, NNODES}) where {RealT, NNODES}
    NNODES
end

@inline get_nodes(basis::LobattoLegendreBasisGPU) = basis.nodes

@inline eachnode(basis::LobattoLegendreBasisGPU) = Base.OneTo(nnodes(basis))

@inline polydeg(basis::LobattoLegendreBasisGPU) = nnodes(basis) - 1

function SolutionAnalyzer(basis::LobattoLegendreBasisGPU;
                          analysis_polydeg = 2 * polydeg(basis))
    RealT = real(basis)
    nnodes_ = analysis_polydeg + 1

    nodes_, weights_ = gauss_lobatto_nodes_weights(nnodes_, RealT)
    vandermonde = polynomial_interpolation_matrix(get_nodes(basis), nodes_)

    # Type conversions to enable possible optimizations of runtime performance 
    # and latency
    nodes = SVector{nnodes_, RealT}(nodes_)
    weights = SVector{nnodes_, RealT}(weights_)

    return LobattoLegendreAnalyzer{RealT, nnodes_, typeof(nodes), typeof(vandermonde)}(nodes,
                                                                                       weights,
                                                                                       vandermonde)
end

# CPU version of LobattoLegendreMortarL2 from Trixi.jl (for reference)
# struct LobattoLegendreMortarL2{RealT <: Real, NNODES,
#                                ForwardMatrix <: AbstractMatrix{RealT},
#                                ReverseMatrix <: AbstractMatrix{RealT}} <: AbstractMortarL2{RealT}
#     forward_upper::ForwardMatrix
#     forward_lower::ForwardMatrix
#     reverse_upper::ReverseMatrix
#     reverse_lower::ReverseMatrix
# end

# Similar to `LobattoLegendreMortarL2` in Trixi.jl but GPU compatible
# struct LobattoLegendreMortarL2GPU{RealT <: Real, NNODES,
#                                   ForwardMatrix <: CuArray{RealT, 2},
#                                   ReverseMatrix <: CuArray{RealT, 2}} <: AbstractMortarL2{RealT}
#     forward_upper::ForwardMatrix
#     forward_lower::ForwardMatrix
#     reverse_upper::ReverseMatrix
#     reverse_lower::ReverseMatrix
# end

# Similar to function `MortarL2` in Trixi.jl
function MortarL2GPU(basis::LobattoLegendreBasisGPU)
    RealT = real(basis)
    nnodes_ = nnodes(basis)

    forward_upper_ = calc_forward_upper(nnodes_, RealT)
    forward_lower_ = calc_forward_lower(nnodes_, RealT)
    reverse_upper_ = calc_reverse_upper(nnodes_, Val(:gauss), RealT)
    reverse_lower_ = calc_reverse_lower(nnodes_, Val(:gauss), RealT)

    # Convert to GPU arrays
    forward_upper = CuArray{RealT}(forward_upper_)
    forward_lower = CuArray{RealT}(forward_lower_)
    reverse_upper = CuArray{RealT}(reverse_upper_)
    reverse_lower = CuArray{RealT}(reverse_lower_)

    LobattoLegendreMortarL2{RealT, nnodes_, typeof(forward_upper),
                            typeof(reverse_upper)}(forward_upper, forward_lower,
                                                   reverse_upper, reverse_lower)
end
