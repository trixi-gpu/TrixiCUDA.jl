module TrixiGPU

# Include other packages that are used in TrixiGPU.jl
# using Reexport: @reexport

using CUDA: @cuda, CuArray, HostKernel, launch_configuration, threadIdx
using Trixi: AbstractEquations

import Trixi: get_node_vars, get_node_coords, get_surface_node_vars

# Include other source files
include("function.jl")
include("auxiliary/auxiliary.jl")

# Export the public APIs
# export configurator_1d, configurator_2d, configurator_3d

end
