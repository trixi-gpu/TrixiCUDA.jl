# TrixiCUDA.jl

TrixiCUDA.jl provides high-order DG solvers for hyperbolic PDEs with native CUDA.jl kernels.

# Performance Benchmarks

Performance is tracked by benchmarking `TrixiCUDA.rhs_hyperbolic!`, the evaluation of the semidiscrete right-hand side, in Float64 on the initial mesh of each case. All cases are adapted from [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) or [TrixiShallowWater.jl](https://github.com/trixi-framework/TrixiShallowWater.jl) examples on `TreeMesh`. The cases are selected so that every right-hand-side code path in each spatial dimension is covered by two to three cases, or by the only available example where no second one exists.

## Feature Coverage

The table lists, for each spatial dimension, how many of the 37 benchmark cases exercise each right-hand-side code path.

| Category | Feature | 1D | 2D | 3D |
|:--|:--|:-:|:-:|:-:|
| **Volume integral** | Weak form | 2 | 3 | 3 |
| | Flux differencing | 3 | 2 | 2 |
| | Flux differencing, nonconservative | 3 | 3 | 3 |
| | Shock capturing | 3 | 3 | 2 |
| | Shock capturing, nonconservative | 2 | 2 | 1 † |
| **Boundaries** | Non-periodic | 2 | 2 | 2 |
| | Non-periodic, nonconservative | 2 | 1 † | n/a |
| **Mesh** | Mortars | n/a | 2 | 2 |
| | Mortars, nonconservative | n/a | 1 † | 1 † |
| **Sources** | Source terms | 3 | 3 | 2 |
| **Total** | | **13 cases** | **13 cases** | **11 cases** |

!!! note "Coverage limits"
    † Covered by a single case because only one suitable `TreeMesh` example exists in Trixi.jl or TrixiShallowwater.jl.

    n/a: Mortars do not occur in 1D, and in 3D the boundary flux uses the same code path for conservative and nonconservative equations.

## Cases by Dimension

The tables below list the individual cases behind the coverage summary, grouped by spatial dimension. For each case, they give the equation system, the right-hand-side features it exercises, and the Trixi.jl or TrixiShallowWater.jl example it is adapted from.

**Case** is the file name, **Equations** is the equation system solved, **Features** lists the right-hand-side code paths the case exercises, and this column uses the following abbreviations:

| Abbreviation | Meaning |
|:--|:--|
| WF | Weak form volume integral |
| FD | Flux differencing volume integral |
| SC | Shock capturing volume integral |
| -nc | With nonconservative terms (e.g. FD-nc, BC-nc) |
| BC | Non-periodic boundary conditions |
| M | Mortars |
| Src | Source terms |

**Degrees / Levels** gives the default polynomial degree and initial refinement level used in CI mini benchmark runs. (Sweep benchmark is also implemented in CI by varying both across a grid of degrees and levels; see [Sweep Benchmarks](sweeps.md).) These match the values in the original Trixi.jl or TrixiShallowWater.jl examples.

### 1D (13 cases)

| Case | Equations | Features | Degrees / Levels |
|:--|:--|:--|:-:|
| `advection_basic` | Linear advection | WF | 3 / 4 |
| `hypdiff_nonperiodic` | Hyperbolic diffusion | WF, BC, Src | 4 / 3 |
| `euler_ec` | Compressible Euler | FD | 3 / 5 |
| `mhd_ec` | Ideal GLM-MHD | FD | 3 / 4 |
| `mhd_alfven_wave` | Ideal GLM-MHD | FD | 4 / 2 |
| `shallowwater_ec` | Shallow water | FD-nc | 4 / 4 |
| `shallowwater_source_terms` | Shallow water | FD-nc, Src | 3 / 3 |
| `euler_quasi_source_terms_dirichlet` | Quasi compressible Euler | FD-nc, BC-nc, Src | 4 / 4 |
| `euler_shock` | Compressible Euler | SC | 3 / 5 |
| `euler_blast_wave` | Compressible Euler | SC | 3 / 6 |
| `mhd_briowu_shock_tube` | Ideal GLM-MHD | SC, BC | 4 / 4 |
| `euler_quasi_discontinuous` | Quasi compressible Euler | SC-nc | 3 / 6 |
| `shallowwater_shock_capturing` | Shallow water | SC-nc, BC-nc | 4 / 3 |

### 2D (13 cases)

| Case | Equations | Features | Degrees / Levels |
|:--|:--|:--|:-:|
| `advection_basic` | Linear advection | WF | 3 / 4 |
| `euler_source_terms_nonperiodic` | Compressible Euler | WF, BC, Src | 3 / 4 |
| `hypdiff_nonperiodic` | Hyperbolic diffusion | WF, BC, Src | 4 / 3 |
| `euler_ec` | Compressible Euler | FD | 3 / 5 |
| `euler_vortex_mortar_split` | Compressible Euler | FD, M | 3 / 4 |
| `mhd_alfven_wave_dirichlet` | Ideal GLM-MHD | FD-nc, BC-nc | 3 / 4 |
| `mhd_alfven_wave_mortar` | Ideal GLM-MHD | FD-nc, M-nc | 3 / 4 |
| `shallowwater_source_terms` | Shallow water | FD-nc, Src | 3 / 3 |
| `euler_shock` | Compressible Euler | SC | 3 / 5 |
| `euler_sedov_blast_wave` | Compressible Euler | SC | 3 / 6 |
| `euler_blob_mortar` | Compressible Euler | SC, M | 3 / 4 |
| `mhd_orszag_tang` | Ideal GLM-MHD | SC-nc | 3 / 4 |
| `mhd_rotor` | Ideal GLM-MHD | SC-nc | 4 / 4 |

### 3D (11 cases)

| Case | Equations | Features | Degrees / Levels |
|:--|:--|:--|:-:|
| `hypdiff_nonperiodic` | Hyperbolic diffusion | WF, BC, Src | 4 / 2 |
| `advection_mortar` | Linear advection | WF, M | 3 / 2 |
| `euler_mortar` | Compressible Euler | WF, M, Src | 3 / 2 |
| `euler_ec` | Compressible Euler | FD | 3 / 3 |
| `euler_taylor_green_vortex` | Compressible Euler | FD | 3 / 3 |
| `mhd_alfven_wave` | Ideal GLM-MHD | FD-nc | 3 / 2 |
| `mhd_ec` | Ideal GLM-MHD | FD-nc | 3 / 2 |
| `mhd_alfven_wave_mortar` | Ideal GLM-MHD | FD-nc, M-nc | 3 / 2 |
| `euler_shock` | Compressible Euler | SC | 3 / 3 |
| `euler_sedov_blast_wave` | Compressible Euler | SC, BC | 3 / 2 |
| `mhd_ec_shockcapturing` | Ideal GLM-MHD | SC-nc | 4 / 3 |