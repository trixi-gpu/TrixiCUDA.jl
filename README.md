# TrixiCUDA.jl

[![Build status (Github Actions)](https://github.com/trixi-gpu/TrixiCUDA.jl/workflows/CI/badge.svg)](https://github.com/trixi-gpu/TrixiCUDA.jl/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![dev docs](https://img.shields.io/badge/docs-dev-orange.svg)](https://trixi-gpu.github.io/TrixiCUDA.jl/dev)

**TrixiCUDA.jl** is a component package of the [**Trixi.jl**](https://github.com/trixi-framework/Trixi.jl) ecosystem and provides GPU acceleration support for solving hyperbolic partial differential equations (PDEs). This package was initialized through the [**Google Summer of Code**](https://summerofcode.withgoogle.com/archive/2023/projects/upstR7K2) program in 2023 and is under active development and testing.

The acceleration focus of this package is currently on the semidiscretization part (with plans to extend to other parts) of the PDE solvers, and [**CUDA.jl**](https://github.com/JuliaGPU/CUDA.jl) is our primary support (will expand to more types of GPUs using [**AMDGPU.jl**](https://github.com/JuliaGPU/AMDGPU.jl), [**OneAPI.jl**](https://github.com/JuliaGPU/oneAPI.jl), and [**Metal.jl**](https://github.com/JuliaGPU/Metal.jl) in the future). 

The complete documentation for this project (and now is available as a package) can be accessed here [**GSoC**](https://huiyuxie.github.io/gsoc23/). Please note that this package is intended to be part of the [**trixi-framework**](https://github.com/trixi-framework) organization in the end. Due to constrained GPU resources and ease of development, it is now under a temporary organization. 

# Package Installation
The package is now in pre-release status and will be registered once the initial release version is published. We want to make sure most key features are ready and optimizations are done before we roll out the first release.

## Users
Users who are interested now can install the package by running the following command in the Julia REPL: 
```julia
julia> using Pkg; Pkg.add(url="https://github.com/trixi-gpu/TrixiCUDA.jl.git")
```
Then the package can be used with the following simple command:
```julia
julia> using TrixiCUDA
```
This package serves as a support package, so it is recommended to these packages together:
```julia
julia> using Trixi, TrixiCUDA, OrdinaryDiffEq
```

## Developers
Developers can start their development by first forking and cloning the repository to their terminal. 

Then enter the Julia REPL in the package directory, activate and instantiate the environment by running the following command:
```julia
julia> using Pkg; Pkg.activate("."); Pkg.instantiate()
```

Please note that GPU-related tests are run locally rather than on CI remotely due to constrained resources. Once the repository is ready to be transferred back to the organization, we are going to set up JuliaGPU CI infrastructure to run tests on a system with a GPU using [**JuliaGPU Buildkite**](https://github.com/JuliaGPU/buildkite).

# Supported Mesh and Solver Types
Our current focus is on the semidiscretization of PDEs. The table below shows the status of this work across different mesh types and solvers. Looking ahead, we plan to extend parallelization to include mesh initialization and callbacks on the GPU. 

| Mesh Type          | Spatial Dimension | Solver Type | Status         |
|--------------------|-------------------|-------------|----------------|
| `TreeMesh`         | 1D, 2D, 3D        | `DGSEM`     | ✅ Supported    |
| `StructuredMesh`   | 1D, 2D, 3D        | `DGSEM`     | 🛠️ In Development|
| `UnstructuredMesh` | 2D                | `DGSEM`     | 🟡 Planned      |
| `P4estMesh`        | 2D, 3D            | `DGSEM`     | 🟡 Planned      |
| `DGMultiMesh`      | 1D, 2D, 3D        | `DGMulti`   | 🟡 Planned      |

# Example of PDE Semidiscretization on GPU
Let's take a look at a simple example to see how to use **TrixiCUDA.jl** to run the simulation on the GPU (now only CUDA-compatible).

```julia
# Take 1D linear advection equation as an example
using Trixi, TrixiCUDA
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
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize_gpu(semi, (0.0, 1.0)) # from TrixiCUDA.jl

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

# Benchmarks
Please check benchmark branch and this part will be updated soon.

# Show Your Support
We always welcome new contributors to join us in future development. Please feel free to reach out if you would like to get involved!
