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
TrixiCUDA.rhs_hyperbolic!
```

## GPU Kernels in Semidiscretization
```@docs
TrixiCUDA.calc_volume_integral!
TrixiCUDA.prolong2interfaces!
TrixiCUDA.calc_interface_flux!
TrixiCUDA.prolong2boundaries!
TrixiCUDA.calc_boundary_flux!
TrixiCUDA.prolong2mortars!
TrixiCUDA.calc_mortar_flux!
TrixiCUDA.calc_surface_integral!
TrixiCUDA.apply_jacobian!
TrixiCUDA.calc_sources!
```
