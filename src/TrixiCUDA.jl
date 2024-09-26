module TrixiCUDA

# Include other packages that are used in TrixiCUDA.jl
# using Reexport: @reexport

using CUDA
using CUDA: @cuda, CuArray, HostKernel,
            threadIdx, blockIdx, blockDim, similar, launch_configuration

using Trixi: AbstractEquations, AbstractContainer,
             InterfaceContainer1D, ElementContainer1D, BoundaryContainer1D,
             True, False,
             TreeMesh, DGSEM,
             BoundaryConditionPeriodic, SemidiscretizationHyperbolic,
             VolumeIntegralWeakForm, VolumeIntegralFluxDifferencing, VolumeIntegralShockCapturingHG,
             LobattoLegendreMortarL2,
             flux, ntuple, nvariables, nnodes, nelements, nmortars,
             local_leaf_cells, init_elements, init_interfaces, init_boundaries, init_mortars,
             wrap_array, compute_coefficients, have_nonconservative_terms,
             boundary_condition_periodic,
             digest_boundary_conditions, check_periodicity_mesh_boundary_conditions,
             set_log_type!, set_sqrt_type!

import Trixi: get_node_vars, get_node_coords, get_surface_node_vars,
              nelements, ninterfaces

using SciMLBase: ODEProblem, FullSpecialize

using StrideArrays: PtrArray

using StaticArrays: SVector

# Include other source files
include("auxiliary/auxiliary.jl")
include("semidiscretization/semidiscretization.jl")
include("solvers/solvers.jl")

# Change to use the Base.log and Base.sqrt - need to be fixed to avoid outputs
set_log_type!("log_Base")
set_sqrt_type!("sqrt_Base")

# Export the public APIs
export SemidiscretizationHyperbolicGPU
export semidiscretizeGPU

end
