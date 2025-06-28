# Rewrite `DGSEM` in Trixi.jl to add the specialized parts of the arrays required to copy the arrays 
# from CPU to GPU.

# Maybe the return type of `DGSEMGPU` should be change?

function DGSEMGPU(basis::LobattoLegendreBasisGPU,
                  surface_flux = flux_central,
                  volume_integral = VolumeIntegralWeakForm(),
                  mortar = MortarL2GPU(basis))
    surface_integral = SurfaceIntegralWeakForm(surface_flux)

    # Since we use `LobattoLegendreBasisGPU`, the type degrades to `DG` if 
    # we create a new `DGSEMGPU` struct.
    return DG{typeof(basis), typeof(mortar), typeof(surface_integral),
              typeof(volume_integral)}(basis, mortar, surface_integral, volume_integral)
end

function DGSEMGPU(basis::LobattoLegendreBasisGPU,
                  surface_integral::AbstractSurfaceIntegral,
                  volume_integral = VolumeIntegralWeakForm(),
                  mortar = MortarL2GPU(basis))

    # Since we use `LobattoLegendreBasisGPU`, the type degrades to `DG` if 
    # we create a new `DGSEMGPU` struct.
    return DG{typeof(basis), typeof(mortar), typeof(surface_integral),
              typeof(volume_integral)}(basis, mortar, surface_integral, volume_integral)
end

"""
    DGSEMGPU(; RealT = Float64,
               polydeg::Integer,
               surface_flux = flux_central,
               surface_integral = SurfaceIntegralWeakForm(surface_flux),
               volume_integral = VolumeIntegralWeakForm()) 

Create a discontinuous Galerkin spectral element method (DGSEM) optimized for GPU computations, 
using a Lobatto-Legendre basis that is partially implemented on the GPU (`LobattoLegendreBasisGPU`).

!!! warning "Experimental implementation"
    This is an experimental implementation and may change or be removed in future releases due to 
    ongoing performance optimizations.
"""
function DGSEMGPU(; RealT = Float64,
                  polydeg::Integer,
                  surface_flux = flux_central,
                  surface_integral = SurfaceIntegralWeakForm(surface_flux),
                  volume_integral = VolumeIntegralWeakForm())
    basis = LobattoLegendreBasisGPU(RealT, polydeg)

    # Since we use `LobattoLegendreBasisGPU`, the type degrades to `DG` if 
    # we create a new `DGSEMGPU` struct.
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
