module TrixiGPU

# Include other packages that are used in TrixiGPU.jl

using CUDA: @cuda, CuArray, HostKernel, launch_configuration, threadIdx
using Trixi: AbstractEquations

# Include other source files

include("function.jl")
include("auxiliary/auxiliary.jl")

# Export the public APIs
export configurator_1d, configurator_2d, configurator_3d

end
