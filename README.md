# TrixiCUDA.jl

[![Build status (Github Actions)](https://github.com/trixi-gpu/TrixiCUDA.jl/workflows/CI/badge.svg)](https://github.com/trixi-gpu/TrixiCUDA.jl/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![dev docs](https://img.shields.io/badge/docs-dev-orange.svg)](https://trixi-gpu.github.io/TrixiCUDA.jl/dev)

TrixiCUDA.jl offers CUDA acceleration for solving hyperbolic PDEs.

Package docs: https://trixi-gpu.github.io/TrixiCUDA.jl/dev (in development) \
General docs: https://trixi-gpu.github.io

> [!WARNING]
> The package may not always be updated with the latest updates in Trixi.jl. Forcing an update of Trixi.jl as a dependency for TrixiCUDA.jl beyond the version bounds specified in Project.toml may cause unexpected errors.

*Update on Jun 6, 2025*:
- The scalar indexing issue on GPU arrays has been fixed for most common examples, but for more complicated cases you can run them error‐free by doing `using CUDA; CUDA.allowscalar(true)`, and a permanent fix will be available soon.

*Update on Mar 19, 2025*:
- The issue between the latest version of CUDA.jl and Trixi.jl has been resolved. The package is now compatible with CUDA.jl v5.7.0 and Trixi.jl v0.10 (see [TrixiCUDA.jl PR #141](https://github.com/trixi-gpu/TrixiCUDA.jl/pull/141)).

*Update on Jan 28, 2025*:
- It is recommended to update your Julia version to 1.10.8 (the latest LTS release) to avoid the issue of circular dependencies during package precompilation, which is present in Julia 1.10.7 (see [issue](https://discourse.julialang.org/t/circular-dependency-warning/123388)).

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

Let's take a look at a simple example to see how to use TrixiCUDA.jl to run the simulation on the GPU.

```julia
# Take 1D linear advection equation as an example
using Trixi, TrixiCUDA
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEMGPU(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0
coordinates_max = 1.0 

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretizeGPU(semi, (0.0, 1.0))

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100)

save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.6)

callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0,
            ode_default_options()..., callback = callbacks);
```
Please also try the examples in the tests directory, as they are always the most up-to-date.

# Benchmarks
Please check out [benchmark.ipynb](https://github.com/trixi-gpu/TrixiCUDA.jl/blob/main/benchmark/benchmark.ipynb) in the [benchmark](https://github.com/trixi-gpu/TrixiCUDA.jl/tree/main/benchmark) directory to run the existing benchmark examples or your own. At present, our focus is on optimizing and benchmarking the semidiscretization process, so the benchmarks mainly measure the performance of `rhs!` functions on CPU and GPU.

See the most recent benchmark results for the semidiscretization process [here](https://trixi-gpu.github.io/benchmark/).


# Join Our Slack Group

[Join Our Slack Group](https://trixi-framework.slack.com/archives/C056B6F4V47)

If you run into any issues or have questions, join our Slack channel [Trixi framework #gpu](https://trixi-framework.slack.com/archives/C056B6F4V47) for support and discussion. Connect with developers and users to troubleshoot problems, share ideas, and collaborate on improvements.
