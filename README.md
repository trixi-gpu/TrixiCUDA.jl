# linear_advection_cuda

- The folder `trixi` stores folders from `Trixi.jl`, specifically `trixi/src`, `trixi/examples`, and `trixi/test`.
- The file `header.jl` can be used as test environment initializer for running tests in other files like `linear_advection_1d.jl`.
- The file `linear_advection_1d.jl` runs tests for prototyping 1D GPU code.
- The file `simple_kernels.jl` creates sample kernels for running 1D, 2D, and 3D GPU code.
- The folder `docs` contains useful resources for this project.
- The folder `profile` contains contents about how to profile GPU kernels.
- The folder `linear_adveciton` contains prework of prototyping GPU code for solving linear advection equations (currently 1D code).

TODO List:
- Simplify kernel arguments in 1D, 2D, and 3D linear advection GPU code
- Complete basic kernels for `rhs!()` from `dg_1d.jl`, `dg_2d.jl`, and `dg_3d.jl`
- Add more features to 1D, 2D, and 3D linear advection kernel functions
