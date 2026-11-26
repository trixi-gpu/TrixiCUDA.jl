# Benchmark Results

## Sweep Heatmaps

We study how the cost of evaluating the semidiscrete right-hand side varies with the polynomial degree p and the refinement level L. Each case is examined along two lines through this plane: a level sweep, which refines the mesh at a fixed degree p = 3, and a degree sweep, which raises p on a fixed mesh with L = 12 in 1D, L = 6 in 2D, and L = 4 in 3D.

At every point, the target revision is compared with its baseline. Both revisions are measured in three alternating processes, and the speedup is defined as the median ratio of their process medians. Its uncertainty is quantified by a 95% two-level bootstrap interval, which accounts for variation both within and between processes.

A change is reported only when it exceeds the noise band of ±3%, the range a speedup can reach through measurement noise alone. This band was calibrated with runs that compare identical code. A cell is shown in green when its entire interval lies above the band, in red when it lies below, and in white when it lies within. Hatched cells mark inconclusive results, whose intervals cross an edge of the band. Hovering over a cell reveals its details.

```@raw html
<div id="sweep-heatmaps"></div>
```
