# TrixiGPU.jl

[![Build Status](https://github.com/huiyuxie/TrixiGPU.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/huiyuxie/TrixiGPU.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**TrixiGPU.jl** is a component package of the [**Trixi.jl**](https://github.com/trixi-framework/Trixi.jl) ecosystem and provides GPU acceleration support for solving hyperbolic partial differential equations (PDEs). This package was initialized through the [**Google Summer of Code**](https://summerofcode.withgoogle.com/archive/2023/projects/upstR7K2) program in 2023 and is still under development.

The acceleration focus of this package is currently on the semidiscretization part (with plans to extend to other parts) of the PDE solvers, and [**CUDA.jl**](https://github.com/JuliaGPU/CUDA.jl) is our primary support (will expand to more types of GPUs using [**AMDGPU.jl**](https://github.com/JuliaGPU/AMDGPU.jl), [**OneAPI.jl**](https://github.com/JuliaGPU/oneAPI.jl), and [**Metal.jl**](https://github.com/JuliaGPU/Metal.jl) in the future). 

Please check the progress of our development [**here**](https://github.com/users/huiyuxie/projects/2).

# Example of Semidiscretization on GPU
```julia
# Take 1D Linear Advection Equation as an example
using Trixi, TrixiGPU
using OrdinaryDiffEq

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0
coordinates_max = 1.0 

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000) # set maximum capacity of tree data structure

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize_gpu(semi, (0.0, 1.0)) # from TrixiGPU.jl

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100)

save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.6)

callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, save_everystep = false, callback = callbacks)

summary_callback()
```

# Supported Mesh and Solver Types
Our current focus is on the semidiscretization of PDEs. The table below shows the status of this work across different mesh types and solvers. Looking ahead, we plan to extend parallelization to include mesh initialization and callbacks on the GPU. 

| Mesh Type          | Spatial Dimension | Solver Type | Status         |
|--------------------|-------------------|-------------|----------------|
| `TreeMesh`         | 1D, 2D, 3D        | `DGSEM`     | ✅ Supported    |
| `StructuredMesh`   | 1D, 2D, 3D        | `DGSEM`     | 🛠️ In Development|
| `UnstructuredMesh` | 2D                | `DGSEM`     | 🟡 Planned      |
| `P4estMesh`        | 2D, 3D            | `DGSEM`     | 🟡 Planned      |
| `DGMultiMesh`      | 1D, 2D, 3D        | `DGMulti`   | 🟡 Planned      |

# GPU Kernels to be Implemented
Kernels left to be implemented on `TreeMesh` with `DGSEM`:
- 1D Kernels: 1) `calc_volume_integral!` - `volume_integral::VolumeIntegralShockCapturingHG`
- 2D Kernels: 1) `calc_volume_integral!` - `volume_integral::VolumeIntegralShockCapturingHG`, 2) `calc_mortar_flux!`
- 3D Kernels: 1) `calc_volume_integral!` - `volume_integral::VolumeIntegralShockCapturingHG`, 2) `calc_mortar_flux!`

# Show Your Support!
We always welcome new people to join us, please feel free to contribute. Also, if you find this package interesting and inspiring, please give it a star. Thanks!
