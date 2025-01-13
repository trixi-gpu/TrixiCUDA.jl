# TrixiCUDA.jl

[![Build status (Github Actions)](https://github.com/trixi-gpu/TrixiCUDA.jl/workflows/CI/badge.svg)](https://github.com/trixi-gpu/TrixiCUDA.jl/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![dev docs](https://img.shields.io/badge/docs-dev-orange.svg)](https://trixi-gpu.github.io/TrixiCUDA.jl/dev)

TrixiCUDA.jl offers CUDA acceleration for solving hyperbolic PDEs.

⚠️ **Warning:** Our package may not always be updated with the latest updates or improvements in Trixi.jl. Forcing an update of Trixi.jl as a dependency for TrixiCUDA.jl beyond the version bounds specified in `Project.toml` may cause unexpected errors.

*Update on Dec 31, 2024*:
- The initial round of kernel optimization starts with the volume integral kernels (see [TrixiCUDA.jl PR #102](https://github.com/trixi-gpu/TrixiCUDA.jl/pull/102)) and will later extend to all existing kernels used in the semidiscretization process. The primary approach includes improving global memory access patterns, using shared memory, and selecting appropriate launch sizes to reduce warp stalls and achieve better occupancy.

*Update on Nov 21, 2024*:
- Due to the [incompatibility issue](https://github.com/trixi-framework/Trixi.jl/issues/2108) from upstream with Trixi.jl and CUDA.jl in Julia v1.11, this package now supports only Julia v1.10. Using or developing this package with Julia v1.11 will result in precompilation errors. To fix this, downgrade to Julia v1.10. If you have any other problems, please file issues [here](https://github.com/trixi-gpu/TrixiCUDA.jl/issues).

*Update on Oct 30, 2024*: 
- The general documentation is now available at https://trixi-gpu.github.io (in development).  
- Documentation specific to this package can be found at https://trixi-gpu.github.io/TrixiCUDA.jl/dev (in development).

[Archived Update](https://trixi-gpu.github.io/update/)


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
Please note that after package precompilation, the terminal will generate some output like:
```
(TrixiCUDA) pkg> precompile
Precompiling project...
        Info Given TrixiCUDA was explicitly requested, output will be shown live
[ Info: Please restart Julia and reload Trixi.jl for the `log` computation change to take effect
[ Info: Please restart Julia and reload Trixi.jl for the `sqrt` computation change to take effect
```
This is normal, and you can safely ignore it (i.e., there is no need to restart Julia and reload Trixi.jl). This occurs because custom `log` and `sqrt` functions from Trixi.jl are currently not supported on GPUs, so we are using the `log` and `sqrt` functions from Julia Base (i.e., Base.jl).

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

⚠️ **Warning:** Due to the cache initialization process being moved to the GPU for performance optimization, most examples may raise errors because of mismatched CPU and GPU APIs. Please try the examples in the tests, as they are always the most up-to-date.

Let's take a look at a simple example to see how to use TrixiCUDA.jl to run the simulation on the GPU.

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

ode = semidiscretizeGPU(semi, (0.0, 1.0)) # from TrixiCUDA.jl

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
Please refer to the benchmark directory to conduct your own benchmarking based on different PDE examples. The official benchmarking report for the semidiscretization process will be released in the future.


# Show Your Support
We always welcome new contributors to join us in future development. Please feel free to reach out if you would like to get involved!
