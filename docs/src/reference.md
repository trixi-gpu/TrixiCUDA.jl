# Public APIs

All APIs currently default to double precision (`Float64`). Single precision (`Float32`) is supported but still experimental, as it still needs testing to avoid type promotion. 

```@docs
TrixiCUDA.LobattoLegendreBasisGPU
TrixiCUDA.DGSEMGPU
TrixiCUDA.SemidiscretizationHyperbolicGPU
TrixiCUDA.semidiscretizeGPU
```

# Private APIs

## Semidiscretization
```@docs
TrixiCUDA.rhs_gpu!
```

## GPU Kernels in Semidiscretization
```@docs
TrixiCUDA.cuda_volume_integral!
TrixiCUDA.cuda_prolong2interfaces!
TrixiCUDA.cuda_interface_flux!
TrixiCUDA.cuda_prolong2boundaries!
TrixiCUDA.cuda_boundary_flux!
TrixiCUDA.cuda_surface_integral!
TrixiCUDA.cuda_jacobian!
TrixiCUDA.cuda_sources!
```