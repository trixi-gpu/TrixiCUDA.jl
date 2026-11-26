# Benchmark Results

## Sweep Heatmaps

A `[benchmark sweep]` run measures every case along two lines through the plane of polynomial degree p and refinement level L: a level sweep at p = 3, and a degree sweep at L = 12 in 1D, L = 6 in 2D, and L = 4 in 3D. Each cell compares the target with the baseline at one point, with the same measurement and statistics as the mini benchmark: three alternating processes per revision, the median ratio of process medians as the speedup, and a 95% two-level bootstrap interval. A cell is colored only when its whole interval lies outside the noise band; hover over a cell for its details.

```@raw html
<div id="sweep-heatmaps"></div>
```
