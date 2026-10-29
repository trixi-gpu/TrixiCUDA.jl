# TrixiCUDA.jl

TrixiCUDA.jl is a component package of the Trixi.jl ecosystem and provides GPU acceleration support for solving hyperbolic partial differential equations (PDEs).

## Trixi preferences

Loading TrixiCUDA does not change Trixi's preferences. With no user overrides,
Trixi uses `log_Trixi_NaN`, `sqrt_Trixi_NaN`, LoopVectorization, and the `:polyester`
CPU threading backend. The default math returns `NaN` for negative floating-point
inputs, avoiding domain exceptions inside CPU worker loops. Trixi's CUDA extension
provides the device implementation of `log` for `Float32` and `Float64`.
The CPU threading backend does not control TrixiCUDA's explicit CUDA kernel launches.

Preferences saved by older TrixiCUDA versions remain in effect. To restore the
defaults in an existing environment, run the following once, then restart Julia:

```julia
using Trixi
Trixi.set_log_type!("log_Trixi_NaN")
Trixi.set_sqrt_type!("sqrt_Trixi_NaN")
Trixi.set_loop_vectorization!(true)
```
