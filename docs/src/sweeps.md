# Sweep Benchmarks

We study how the cost of evaluating the semidiscrete right-hand side varies with the polynomial degree p and the refinement level L.

## Sweep Heatmaps

Each case is examined along two lines through this plane: a level sweep that refines the mesh at p = 3, and a degree sweep that raises p on a fixed mesh (L = 12 in 1D, 6 in 2D, and 4 in 3D).

At each point, the target is compared with its baseline. Both are measured in five alternating processes; the speedup is the median ratio of their process medians, with a 95% two-level bootstrap interval.

A cell is green when its whole interval lies above the ±3% noise band, red when it lies below, and white when it lies within; hatched cells are inconclusive. The band was calibrated with runs that compare identical code. Hover over a cell for details.

### Latest Sweep

```@raw html
<div id="sweep-heatmaps"></div>
```

!!! note "Latest results"
    The heatmaps on this page show the most recent benchmark sweep: the latest measured revision compared with its baseline. They are updated with every new sweep and track the effect of our ongoing work on speeding up the GPU right-hand side function.