# Rewrite `DGSEM` in Trixi.jl to add the specialized parts of the arrays 
# required to copy the arrays from CPU to GPU.

# Similar to `DGSEM` in Trixi.jl 
function DGSEMGPU(basis::LobattoLegendreBasis,
                  surface_integral::AbstractSurfaceIntegral,
                  volume_integral = VolumeIntegralWeakForm(),
                  mortar = MortarL2GPU(basis))
    return DG{typeof(basis), typeof(mortar), typeof(surface_integral),
              typeof(volume_integral)}(basis, mortar, surface_integral, volume_integral)
end

function DGSEMGPU(; RealT = Float64, # how about setting the default to Float32?
                  polydeg::Integer,
                  surface_flux = flux_central,
                  surface_integral = SurfaceIntegralWeakForm(surface_flux),
                  volume_integral = VolumeIntegralWeakForm())
    basis = LobattoLegendreBasisGPU(RealT, polydeg)

    return DGSEMGPU(basis, surface_integral, volume_integral)
end
