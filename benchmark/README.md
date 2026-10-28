# GPU RHS benchmarks

The scripts in this directory compare CPU and GPU RHS evaluation for individual
problems. `ci/` provides automated comparisons of two TrixiCUDA commits, covering
all 26 existing problem configurations: advection, Euler, hyperbolic diffusion,
MHD, and shallow water, with 1D/2D/3D, mortar, shock-capturing, source-term, and
nonperiodic-boundary cases where available. It benchmarks the complete
`TrixiCUDA.rhs_gpu!`, including its CPU work and synchronization, rather than
individual kernels or an ODE solve.

## Inputs and timing

CI uses Float32 and polynomial degrees/meshes/fluxes matching the existing scripts.
`u` is generated on the host using `StableRNG(123)` and then copied to the GPU.
Euler and MHD use random primitive variables with positive density and pressure,
converted to conservative variables; shallow water uses positive water depth.
No initial condition is evolved in time. These random fields exercise the RHS but
are not intended to reproduce the flow or shock distribution of an ODE solution.

`du` starts at zero and is reset to zero before **each** sample. Two synchronized
warmups compile the RHS and initialize caches; a finite-output check rejects
invalid inputs or failed computations. Timing uses `@benchmark CUDA.@sync`, with
`evals=1`. Random input generation, host-to-device transfers, warmups, and the
synchronized reset are outside the timed region. Each case defaults to at most
100 samples or 2 seconds. The report compares medians; artifacts retain every
sample and the minimum as well.

## CI comparisons

The Buildkite benchmark step uses `queue=cuda`, without restricting the GPU model
or requiring a dedicated benchmark-agent tag. Both revisions run sequentially in
the same job on the same GPU, with Julia 1.10 and one CPU thread. Benchmark jobs
for this repository are serialized. The report records the actual GPU model;
timings from separate builds may come from different hardware and should not be
compared directly.

- PR build: fetch and compare the target-branch tip against the checked-out PR
  commit, recording both exact SHAs.
- Branch push without PR metadata: fetch and compare the pipeline's default
  branch (`main`) against the checked-out branch commit. An open GitHub PR does
  not guarantee that Buildkite supplies PR metadata for a push build.
- Default-branch build: compare the first parent against the new commit. This measures
  the change since the previous baseline, which advances naturally on each
  push; it also handles merge and squash commits. A push containing multiple
  commits compares only its last commit with its first parent.
- All comparisons use the **target's harness**, even if the baseline has no benchmark suite.
  The target environment is resolved first; its manifest is copied to the
  baseline environment, where only the local TrixiCUDA source path is replaced
  with `Pkg.PRESERVE_ALL`. If the two commits cannot use the same dependency
  resolution, comparison fails instead of silently comparing different stacks.

The Markdown report is published as a **Buildkite build annotation** and artifact,
not a PR comment. TOML artifacts contain raw samples, shapes, seeds, commit SHAs,
Julia/GPU/driver/runtime information, and resolved package versions. Both
`Project.toml` and `Manifest.toml` files are attached for reproduction. The report
rejects mismatched environments, dependencies, parameters, or inputs.

Reports are informational: no timing regression threshold is enforced, and an
infrastructure/benchmark failure is a soft failure. `[skip benchmarks]` in the
build message skips the job. Existing correctness tests remain separate.

For comparison, [Trixi.jl uses PkgBenchmark to compare revisions and export Markdown](https://github.com/trixi-framework/Trixi.jl/blob/main/benchmark/run_benchmarks.jl).
This suite uses explicit worktrees so both revisions can share identical random
inputs, benchmark definitions, and resolved dependencies on one GPU, including
when the baseline predates this infrastructure. It needs no publishing token.

## Running locally

From the repository root on a machine with a CUDA GPU:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --threads=1 --project=benchmark benchmark/ci/run.jl benchmark/results.toml "$(git rev-parse HEAD)"
```

To compare with the first parent exactly as a non-PR CI job does:

```sh
bash .buildkite/benchmark.sh
```

Optional environment variables (apply equally to both revisions):

- `TRIXICUDA_BENCHMARK_FILTER=advection_basic_1d`: substring to select cases.
- `TRIXICUDA_BENCHMARK_SAMPLES=10`: maximum samples per case.
- `TRIXICUDA_BENCHMARK_SECONDS=0.2`: time budget per case.

CPU-only validation of report calculations and case/input setup:

```sh
julia benchmark/ci/test_report.jl
julia --threads=1 --project=benchmark benchmark/ci/test_cases.jl
```

Actual GPU smoke test (the two warmups also check finiteness):

```sh
TRIXICUDA_BENCHMARK_FILTER=advection_basic_1d TRIXICUDA_BENCHMARK_SAMPLES=2 \
  julia --threads=1 --project=benchmark benchmark/ci/run.jl benchmark/smoke.toml "$(git rev-parse HEAD)"
```
