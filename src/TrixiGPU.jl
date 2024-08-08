module TrixiGPU

# Include other packages that are used in TrixiGPU.jl (? reorder)
# using Reexport: @reexport

using CUDA: @cuda, CuArray, HostKernel,
            threadIdx, blockIdx, blockDim, similar,
            launch_configuration
using Trixi: AbstractEquations, TreeMesh, VolumeIntegralWeakForm, DGSEM,
             flux, ntuple, nvariables

import Trixi: get_node_vars, get_node_coords, get_surface_node_vars

using StrideArrays: PtrArray

using StaticArrays: SVector

# Include other source files
include("function.jl")
include("auxiliary/auxiliary.jl")
include("solvers/solvers.jl")

# Export the public APIs
# export configurator_1d, configurator_2d, configurator_3d

end
