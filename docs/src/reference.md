# Public APIs

All APIs currently default to double precision (`Float64`). Support for single precision (`Float32`) is considered experimental, as it has not yet undergone comprehensive testing and validation.

```@docs
TrixiCUDA.LobattoLegendreBasisGPU
TrixiCUDA.DGSEMGPU
TrixiCUDA.SemidiscretizationHyperbolicGPU
TrixiCUDA.semidiscretizeGPU
```

# GPU Kernels in Semidiscretizations

All GPU kernels are encapsulated within the semidiscretization (i.e., `rhs_gpu!` function).

```@docs
TrixiCUDA.cuda_volume_integral!
```