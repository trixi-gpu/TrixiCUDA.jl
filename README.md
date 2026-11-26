# TrixiCUDA.jl

[![Buildkite](https://badge.buildkite.com/3c4c88284c3962a32ed073e1f3a90f5fdc32e6df34cf68d527.svg?branch=main)](https://buildkite.com/julialang/trixicuda-dot-jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![dev docs](https://img.shields.io/badge/docs-dev-orange.svg)](https://trixi-gpu.github.io/TrixiCUDA.jl/dev)
[![Julia 1.10+](https://img.shields.io/badge/julia-1.10%2B-9558B2?logo=julia&logoColor=white)](https://github.com/trixi-gpu/TrixiCUDA.jl/blob/main/Project.toml)
[![CUDA accelerated](https://img.shields.io/badge/CUDA-accelerated-76B900?logo=nvidia&logoColor=white)](https://github.com/JuliaGPU/CUDA.jl)

High-order DG solvers for hyperbolic PDEs with native CUDA.jl kernels.

Package docs: https://trixi-gpu.github.io/TrixiCUDA.jl/dev (in development) \
General docs: https://trixi-gpu.github.io

> [!IMPORTANT]
> TrixiCUDA.jl is actively maintained to stay close to the latest releases of Trixi.jl and CUDA.jl. Automated monitoring and AI-assisted maintenance help detect upstream compatibility issues and address them promptly, including breaking changes and compatibility issues introduced by major (`X.0.0`), minor (`X.Y.0`), or patch (`X.Y.Z`) updates. See [Project.toml](Project.toml) for the currently supported dependency versions.
>
> We are also tracking safeguards for AI-assisted changes in [#236](https://github.com/trixi-gpu/TrixiCUDA.jl/issues/236) and will keep regular local code backups as a recovery option.

*Update on Nov 1, 2026*:
- Corrected the state order in 2D/3D nonconservative mortar fluxes to match Trixi.jl v0.17.19, fixing the resulting CPU/GPU discrepancies in MHD mortar tests (see [TrixiCUDA.jl PR #235](https://github.com/trixi-gpu/TrixiCUDA.jl/pull/235)).

*Update on Oct 29, 2026*:
- Loading TrixiCUDA.jl now preserves Trixi.jl's default preferences and any user settings, without overwriting them or printing restart messages (see [TrixiCUDA.jl PR #230](https://github.com/trixi-gpu/TrixiCUDA.jl/pull/230)).

*Update on Oct 28, 2026*:
- Restored 1D/2D shallow-water test coverage using TrixiShallowWater.jl, with CPU/GPU comparisons and scalar indexing disabled. Test dependencies now live in a separate test environment (see [TrixiCUDA.jl PR #226](https://github.com/trixi-gpu/TrixiCUDA.jl/pull/226)).

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
using OrdinaryDiffEqLowStorageRK

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

solver = DGSEMGPU(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0
coordinates_max = 1.0 

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = true)

semi = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition_convergence_test,
                                       solver, boundary_conditions = boundary_condition_periodic)

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
We benchmark `rhs_hyperbolic!`, which evaluates the time derivative of the spatially discretized solution, to guide the optimization of the GPU semidiscretization and to track its performance over time. The 37 benchmark cases cover every right-hand-side code path in 1D, 2D, and 3D; the [documentation](https://trixi-gpu.github.io/TrixiCUDA.jl/dev/) lists each case and the code paths it exercises.

## Running Benchmarks in CI
Benchmarks run on a self-hosted GPU only when the title of a pushed commit asks for them, on `main` or on a branch of an open, non-draft pull request in this repository. Runs are queued, so each has exclusive access to the GPU.

| Marker | What is measured | Where the results appear |
|:--|:--|:--|
| `[benchmark mini]` | Each case at its default polynomial degree and mesh | A PR comment linking the full report in the job summary |
| `[benchmark sweep]` | Each case along a level sweep at p = 3 and a degree sweep on a fixed mesh, 558 points in total | Heatmaps on the [Sweep Benchmarks](https://trixi-gpu.github.io/TrixiCUDA.jl/dev/sweeps/) page of the docs, and a PR comment linking them |

The markers are independent, and a title may contain both. A squash merge whose title contains `[benchmark sweep]` measures the merged commit on `main` and updates the heatmaps in the development docs.

## Methodology
Each run compares two commits on the same GPU:
- **Target:** the pushed commit.
- **Baseline:** the merge base with the PR's base branch, or on `main`, the parent of the pushed commit.

Both commits are measured with the target's benchmark harness, each with its own source code and resolved dependencies, in three alternating Julia processes (baseline, target, baseline, …). The speedup is the median ratio of the baseline and target process medians, with a 95% two-level bootstrap interval. A result is reported as faster or slower only when its whole interval lies outside a ±3% noise band, calibrated with runs that compare identical code; otherwise it is reported as no change or inconclusive.

Benchmarks are informational: they never fail a check, and no regression threshold is enforced. Each run keeps its full report, raw timing samples, and resolved Julia environments as workflow artifacts for 90 days.

## Reference Benchmarks (2025)
For a comprehensive CPU/GPU comparison of semidiscretization performance, see the [2025 benchmark report](https://trixi-gpu.github.io/benchmark/). These results reflect the package versions and test environment used at the time; ongoing performance changes are tracked by the CI benchmarks above.

The benchmarks presented at JuliaCon 2025 can be reproduced with the [benchmark repository](https://github.com/trixi-gpu/juliacon-2025-semidiscretization-benchmark).

# Join Our Slack Group

[Join Our Slack Group](https://trixi-framework.slack.com/archives/C056B6F4V47)

If you run into any issues or have questions, join our Slack channel [Trixi framework #gpu](https://trixi-framework.slack.com/archives/C056B6F4V47) for support and discussion. Connect with developers and users to troubleshoot problems, share ideas, and collaborate on improvements.
