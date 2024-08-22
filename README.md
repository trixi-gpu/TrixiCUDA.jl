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

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0
coordinates_max = 1.0

mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level = 4,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

tspan = (0.0, 1.0)

ode = semidiscretize_gpu(semi, tspan) # from TrixiGPU.jl

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, save_everystep = false, callback = callbacks)
summary_callback()
```

# GPU Kernels to be Implemented
- 1D Kernels: 1) `calc_volume_integral!()` - `volume_integral::VolumeIntegralShockCapturingHG`
- 2D Kernels: 1) `calc_volume_integral!()` - `volume_integral::VolumeIntegralShockCapturingHG`, 2) `calc_mortar_flux!()`
- 3D Kernels: 1) `calc_volume_integral!()` - `volume_integral::VolumeIntegralShockCapturingHG`, 2) `prolong2mortars!()`, 3) `calc_mortar_flux!()` 

# Show Your Support!
We always welcome new people to join us, please feel free to contribute. Also, if you find this package interesting and inspiring, please give it a star. Thanks!
