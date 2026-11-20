# TrixiCUDA.jl

TrixiCUDA.jl provides high-order DG solvers for hyperbolic PDEs with native CUDA.jl kernels.

# Performance Benchmarks

Performance is tracked by benchmarking `TrixiCUDA.rhs_hyperbolic!`, the evaluation of the semidiscrete right-hand side, in Float64 on the initial mesh of each case. All cases are adapted from [Trixi.jl](https://github.com/trixi-framework/Trixi.jl) or
[TrixiShallowWater.jl](https://github.com/trixi-framework/TrixiShallowWater.jl) examples on
`TreeMesh`. The cases are selected so that every right-hand-side code path in each spatial
dimension is covered by two to three cases, or by the only available example where no
second one exists.

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

The **Features** column uses the following abbreviations:

| Abbreviation | Meaning |
|:--|:--|
| WF | Weak form volume integral |
| FD | Flux differencing volume integral |
| SC | Shock capturing volume integral |
| -nc | With nonconservative terms (e.g. FD-nc, BC-nc) |
| BC | Non-periodic boundary conditions |
| M | Mortars |
| Src | Source terms |

### 1D (13 cases)

| Case | Equations | Features | Source |
|:--|:--|:--|:--|
| `advection_basic` | Linear advection | WF | [`elixir_advection_basic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_advection_basic.jl) |
| `hypdiff_nonperiodic` | Hyperbolic diffusion | WF, BC, Src | [`elixir_hypdiff_nonperiodic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_hypdiff_nonperiodic.jl) |
| `euler_ec` | Compressible Euler | FD | [`elixir_euler_ec.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_euler_ec.jl) |
| `mhd_ec` | Ideal GLM-MHD | FD | [`elixir_mhd_ec.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_mhd_ec.jl) |
| `mhd_alfven_wave` | Ideal GLM-MHD | FD | [`elixir_mhd_alfven_wave.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_mhd_alfven_wave.jl) |
| `shallowwater_ec` | Shallow water | FD-nc | [`elixir_shallowwater_ec.jl`](https://github.com/trixi-framework/TrixiShallowWater.jl/blob/v0.3.3/examples/tree_1d_dgsem/elixir_shallowwater_ec.jl) |
| `shallowwater_source_terms` | Shallow water | FD-nc, Src | [`elixir_shallowwater_source_terms.jl`](https://github.com/trixi-framework/TrixiShallowWater.jl/blob/v0.3.3/examples/tree_1d_dgsem/elixir_shallowwater_source_terms.jl) |
| `euler_quasi_1d_source_terms_dirichlet` | Quasi-1D compressible Euler | FD-nc, BC-nc, Src | [`elixir_euler_quasi_1d_source_terms_dirichlet.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_euler_quasi_1d_source_terms_dirichlet.jl) |
| `euler_shock` | Compressible Euler | SC | [`elixir_euler_shockcapturing.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_euler_shockcapturing.jl) |
| `euler_blast_wave` | Compressible Euler | SC | [`elixir_euler_blast_wave.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_euler_blast_wave.jl) |
| `mhd_briowu_shock_tube` | Ideal GLM-MHD | SC, BC | [`elixir_mhd_briowu_shock_tube.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_mhd_briowu_shock_tube.jl) |
| `euler_quasi_1d_discontinuous` | Quasi-1D compressible Euler | SC-nc | [`elixir_euler_quasi_1d_discontinuous.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_euler_quasi_1d_discontinuous.jl) |
| `shallowwater_shock_capturing` | Shallow water | SC-nc, BC-nc | [`elixir_shallowwater_shock_capturing.jl`](https://github.com/trixi-framework/TrixiShallowWater.jl/blob/v0.3.3/examples/tree_1d_dgsem/elixir_shallowwater_shock_capturing.jl) |

### 2D (13 cases)

| Case | Equations | Features | Source |
|:--|:--|:--|:--|
| `advection_basic` | Linear advection | WF | [`elixir_advection_basic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_advection_basic.jl) |
| `euler_source_terms_nonperiodic` | Compressible Euler | WF, BC, Src | [`elixir_euler_source_terms_nonperiodic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_euler_source_terms_nonperiodic.jl) |
| `hypdiff_nonperiodic` | Hyperbolic diffusion | WF, BC, Src | [`elixir_hypdiff_nonperiodic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_hypdiff_nonperiodic.jl) |
| `euler_ec` | Compressible Euler | FD | [`elixir_euler_ec.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_euler_ec.jl) |
| `euler_vortex_mortar_split` | Compressible Euler | FD, M | [`elixir_euler_vortex_mortar_split.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_euler_vortex_mortar_split.jl) |
| `mhd_alfven_wave_dirichlet` | Ideal GLM-MHD | FD-nc, BC-nc | [`elixir_mhd_alfven_wave_dirichlet.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_mhd_alfven_wave_dirichlet.jl) |
| `mhd_alfven_wave_mortar` | Ideal GLM-MHD | FD-nc, M-nc | [`elixir_mhd_alfven_wave_mortar.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_mhd_alfven_wave_mortar.jl) |
| `shallowwater_source_terms` | Shallow water | FD-nc, Src | [`elixir_shallowwater_source_terms.jl`](https://github.com/trixi-framework/TrixiShallowWater.jl/blob/v0.3.3/examples/tree_2d_dgsem/elixir_shallowwater_source_terms.jl) |
| `euler_shock` | Compressible Euler | SC | [`elixir_euler_shockcapturing.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_euler_shockcapturing.jl) |
| `euler_sedov_blast_wave` | Compressible Euler | SC | [`elixir_euler_sedov_blast_wave.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_euler_sedov_blast_wave.jl) |
| `euler_blob_mortar` | Compressible Euler | SC, M | [`elixir_euler_blob_mortar.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_euler_blob_mortar.jl) |
| `mhd_orszag_tang` | Ideal GLM-MHD | SC-nc | [`elixir_mhd_orszag_tang.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_mhd_orszag_tang.jl) |
| `mhd_rotor` | Ideal GLM-MHD | SC-nc | [`elixir_mhd_rotor.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_mhd_rotor.jl) |

### 3D (11 cases)

| Case | Equations | Features | Source |
|:--|:--|:--|:--|
| `hypdiff_nonperiodic` | Hyperbolic diffusion | WF, BC, Src | [`elixir_hypdiff_nonperiodic.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_hypdiff_nonperiodic.jl) |
| `advection_mortar` | Linear advection | WF, M | [`elixir_advection_mortar.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_advection_mortar.jl) |
| `euler_mortar` | Compressible Euler | WF, M, Src | [`elixir_euler_mortar.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_euler_mortar.jl) |
| `euler_ec` | Compressible Euler | FD | [`elixir_euler_ec.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_euler_ec.jl) |
| `euler_taylor_green_vortex` | Compressible Euler | FD | [`elixir_euler_taylor_green_vortex.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_euler_taylor_green_vortex.jl) |
| `mhd_alfven_wave` | Ideal GLM-MHD | FD-nc | [`elixir_mhd_alfven_wave.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_mhd_alfven_wave.jl) |
| `mhd_ec` | Ideal GLM-MHD | FD-nc | [`elixir_mhd_ec.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_mhd_ec.jl) |
| `mhd_alfven_wave_mortar` | Ideal GLM-MHD | FD-nc, M-nc | [`elixir_mhd_alfven_wave_mortar.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_mhd_alfven_wave_mortar.jl) |
| `euler_shock` | Compressible Euler | SC | [`elixir_euler_shockcapturing.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_euler_shockcapturing.jl) |
| `euler_sedov_blast_wave` | Compressible Euler | SC, BC | [`elixir_euler_sedov_blast_wave.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_euler_sedov_blast_wave.jl) |
| `mhd_ec_shockcapturing` | Ideal GLM-MHD | SC-nc | [`elixir_mhd_ec_shockcapturing.jl`](https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_mhd_ec_shockcapturing.jl) |