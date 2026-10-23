# Everything related to Lobatto-Legendre basis adapted for initialization on the GPU.

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
    derivative_hat::DerivativeMatrix
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
    basis_cpu = LobattoLegendreBasis(RealT, polydeg)
    nnodes_ = nnodes(basis_cpu)
    nodes_, weights_ = get_nodes(basis_cpu), basis_cpu.weights
    inverse_weights_ = basis_cpu.inverse_weights
    inverse_vandermonde_legendre_ = basis_cpu.inverse_vandermonde_legendre

    # Lobatto nodes include the endpoints. Retain the weighted boundary matrix
    # for the current GPU surface kernels.
    # TODO: Follow upstream by using `inverse_weights` directly and removing this matrix.
    boundary_interpolation_ = zeros(RealT, nnodes_, 2)
    boundary_interpolation_[1, 1] = first(inverse_weights_)
    boundary_interpolation_[end, 2] = last(inverse_weights_)

    derivative_matrix_ = basis_cpu.derivative_matrix
    derivative_split_ = basis_cpu.derivative_split
    derivative_hat_ = basis_cpu.derivative_hat

    # Convert to GPU arrays
    inverse_weights = CuArray{RealT}(inverse_weights_)

    derivative_matrix = CuArray{RealT}(derivative_matrix_)
    derivative_split = CuArray{RealT}(derivative_split_)
    derivative_split_transpose = CuArray{RealT}(derivative_split_')
    derivative_hat = CuArray{RealT}(derivative_hat_)

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
                                                              derivative_hat)
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
