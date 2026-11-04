# TODO: Support Trixi's generalized HG volume integrals.
# Current GPU kernels require identical default and blended high-order
# flux-differencing operators, combined with first-order subcell FV.
# Upstream: https://github.com/trixi-framework/Trixi.jl/pull/2802

# Rewrite `DGSEM` in Trixi.jl to add the specialized parts of the arrays required to copy the arrays 
# from CPU to GPU.

# Construct a DG solver using this package's GPU basis and kernels.
function DGSEMGPU(basis::LobattoLegendreBasisGPU,
                  surface_flux = flux_central,
                  volume_integral = VolumeIntegralWeakForm(),
                  mortar = MortarL2GPU(basis))
    surface_integral = SurfaceIntegralWeakForm(surface_flux)

    # Preserve the GPU basis and mortar types in the DG solver.
    return DGSEM{typeof(basis), typeof(mortar), typeof(surface_integral),
                 typeof(volume_integral)}(basis, mortar, surface_integral, volume_integral)
end
function DGSEMGPU(basis::LobattoLegendreBasisGPU,
                  surface_integral::AbstractSurfaceIntegral,
                  volume_integral = VolumeIntegralWeakForm(),
                  mortar = MortarL2GPU(basis))
    # Preserve the GPU basis and mortar types in the DG solver.
    return DGSEM{typeof(basis), typeof(mortar), typeof(surface_integral),
                 typeof(volume_integral)}(basis, mortar, surface_integral, volume_integral)
end

"""
    DGSEMGPU(; RealT = Float64,
               polydeg::Integer,
               basis_type = LobattoLegendreBasisGPU,
               surface_flux = flux_central,
               surface_integral = SurfaceIntegralWeakForm(surface_flux),
               volume_integral = VolumeIntegralWeakForm()) 

Create a discontinuous Galerkin spectral element method (DGSEM) optimized for GPU computations, 
using a Lobatto-Legendre basis that is partially implemented on the GPU (`LobattoLegendreBasisGPU`).
Currently, only the `LobattoLegendreBasisGPU` type is supported;
other bases are not supported by the CUDA kernels.

Supported volume integrals are `VolumeIntegralWeakForm`, `VolumeIntegralFluxDifferencing`,
and classic `VolumeIntegralShockCapturingHG`: identical default/blended high-order
flux-differencing operators with first-order subcell FV. Generalized HG configurations
with a different default operator or second-order FV are not yet supported.

!!! warning "Experimental implementation"
    This is an experimental implementation and may change or be removed in future releases due to 
    ongoing performance optimizations.
"""
function DGSEMGPU(; RealT = Float64,
                  polydeg::Integer,
                  basis_type = LobattoLegendreBasisGPU,
                  surface_flux = flux_central,
                  surface_integral = SurfaceIntegralWeakForm(surface_flux),
                  volume_integral = VolumeIntegralWeakForm())
    basis = basis_type(RealT, polydeg)
    if !(basis isa LobattoLegendreBasisGPU)
        throw(ArgumentError("DGSEMGPU requires basis_type to return a LobattoLegendreBasisGPU"))
    end
    # Preserve the GPU basis and mortar types in the DG solver.
    return DGSEMGPU(basis, surface_integral, volume_integral)
end

# Same as `integrate` in Trixi.jl but with GPU signature
# Is there any better way to do this?
function integrate(f, u, basis::LobattoLegendreBasisGPU)
    (; weights) = basis

    res = zero(f(first(u)))
    for i in eachindex(u, weights)
        res += f(u[i]) * weights[i]
    end
    return res
end
