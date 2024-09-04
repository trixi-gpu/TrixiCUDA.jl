# Provide GPU Support for Trixi.jl via CUDA.jl
This project is undertaken as part of the [Google Summer of Code 2023](https://summerofcode.withgoogle.com/) program and is in the developing and testing phase. Please check the [Project Summary](https://gist.github.com/huiyuxie/44b561f9f854aada98fdb37036081454) for future steps. 

## Project Directory Structure
- The folder `trixi` stores folders from `Trixi.jl`, specifically `trixi/src`, `trixi/examples`, and `trixi/test`.
- The folder `docs` contains useful resources and instructions for this project.
- The folder `profile` contains contents about how to profile GPU kernels.
- The folder `src` contains primary files (`cuda_dg_1d.jl`, `cuda_dg_2d.jl`, and `cuda_dg_3d.jl`) that are used to test the performance of `rhs!` functions implemented in CUDA.jl.

## Kernels to be Implemented
- 1D Kernels: 1) `calc_volume_integral!` - `volume_integral::VolumeIntegralShockCapturingHG`
- 2D Kernels: 1) `calc_volume_integral!` - `volume_integral::VolumeIntegralShockCapturingHG`, 2) `calc_mortar_flux!`
- 3D Kernels: 1) `calc_volume_integral!` - `volume_integral::VolumeIntegralShockCapturingHG`, 2) `prolong2mortars!`, 3) `calc_mortar_flux!` 
