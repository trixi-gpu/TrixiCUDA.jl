# Public APIs

All APIs currently default to double precision (`Float64`). Single precision (`Float32`) is supported but still experimental, as it still needs testing to avoid type promotion. 

```@docs
TrixiCUDA.LobattoLegendreBasisGPU
TrixiCUDA.DGSEMGPU
TrixiCUDA.SemidiscretizationHyperbolicGPU
TrixiCUDA.semidiscretizeGPU
```

# Semidiscretization


# GPU Kernels in Semidiscretization

All GPU kernels are encapsulated within the semidiscretization (i.e., `rhs_gpu!` function).

```@docs
TrixiCUDA.cuda_volume_integral!
```