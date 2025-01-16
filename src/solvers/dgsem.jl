# Rewrite `DGSEM` in Trixi.jl to add the specialized parts of the arrays 
# required to copy the arrays from CPU to GPU.

function DGSEMGPU(; RealT = Float64, # how about setting the default to Float32?
                  polydeg::Integer,
                  surface_flux = flux_central,
                  surface_integral = SurfaceIntegralWeakForm(surface_flux),
                  volume_integral = VolumeIntegralWeakForm())
    basis = LobattoLegendreBasis(RealT, polydeg)

    return DGSEM(basis, surface_integral, volume_integral)
end
