# Public APIs

All APIs currently default to double precision (`Float64`). Single precision (`Float32`) is supported but still experimental, as it still needs testing to avoid type promotion. 

```@docs
TrixiCUDA.LobattoLegendreBasisGPU
TrixiCUDA.DGSEMGPU
TrixiCUDA.SemidiscretizationHyperbolicGPU
TrixiCUDA.semidiscretizeGPU
```

## Compatibility with Trixi

- Pass `boundary_conditions` explicitly to `SemidiscretizationHyperbolicGPU`, including
  `boundary_condition_periodic` on periodic meshes.
- `DGSEMGPU` accepts `basis_type(RealT, polydeg)`, which must return a
  `LobattoLegendreBasisGPU`. Other bases are not supported by the CUDA kernels.
- `semidiscretizeGPU` converts `tspan` to `real(semi)` before evaluating the initial condition.
  It accepts Trixi's `jac_prototype` and `colorvec` options and converts the prototype's element
  type to that of the solution. The prototype's storage backend is preserved; GPU differentiation
  and implicit time stepping still depend on the selected SciML solver.
- Supported volume integrals are weak form, flux differencing, and classic HG shock capturing.
  For HG, the default and blended high-order operators must be identical flux-differencing
  operators, paired with first-order subcell FV. The usual `VolumeIntegralShockCapturingHG`
  constructor provides this configuration. Generalized HG with a different default operator,
  second-order FV, and standalone `VolumeIntegralPureLGLFiniteVolume` are not supported.
  The HG indicator is still evaluated on the CPU.

# Private APIs

These functions use the corresponding Trixi names and argument order in the `TrixiCUDA`
namespace. Backend arguments use `CUDA.CUDABackend()`; as in Trixi, the 1D interface methods
and TreeMesh mortar methods do not take a backend. The volume integral and RHS retain an
additional `cache_cpu` argument for the host-side HG indicator. Boundary traits and empty
mortar/boundary checks are handled internally.

GPU tree containers use `TreeElementContainerGPU1D`, `TreeInterfaceContainerGPU1D`, and
`TreeBoundaryContainerGPU1D`, with corresponding 2D/3D types and
`TreeL2MortarContainerGPU2D`/`TreeL2MortarContainerGPU3D`.

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
