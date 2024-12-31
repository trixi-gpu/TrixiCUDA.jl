module TrixiCUDA

# Include other packages that are used in TrixiCUDA.jl
# using Reexport: @reexport

using CUDA
using CUDA: @cuda, CuArray, HostKernel,
            threadIdx, blockIdx, blockDim, similar, launch_configuration

using Trixi: AbstractEquations, AbstractContainer, AbstractMesh, AbstractSemidiscretization,
             True, False, TreeMesh, DGSEM, SemidiscretizationHyperbolic,
             ElementContainer1D, ElementContainer2D, ElementContainer3D,
             InterfaceContainer1D, InterfaceContainer2D, InterfaceContainer3D,
             BoundaryContainer1D, BoundaryContainer2D, BoundaryContainer3D,
             LobattoLegendreMortarL2, L2MortarContainer2D, L2MortarContainer3D,
             BoundaryConditionPeriodic, BoundaryConditionDirichlet,
             VolumeIntegralWeakForm, VolumeIntegralFluxDifferencing, VolumeIntegralShockCapturingHG,
             allocate_coefficients, mesh_equations_solver_cache,
             flux, ntuple, nvariables, nnodes, nelements, nmortars,
             local_leaf_cells, init_elements, init_interfaces, init_boundaries, init_mortars,
             wrap_array, compute_coefficients,
             have_nonconservative_terms, boundary_condition_periodic,
             digest_boundary_conditions, check_periodicity_mesh_boundary_conditions,
             set_log_type!, set_sqrt_type!

import Trixi: get_node_vars, get_node_coords, get_surface_node_vars,
              nelements, ninterfaces, nmortars

using SciMLBase: ODEProblem, FullSpecialize

using StaticArrays: SVector

# Change to use the Base.log and Base.sqrt 
# FIXME: Need to be fixed to avoid precompilation outputs
set_log_type!("log_Base")
set_sqrt_type!("sqrt_Base")

# Include other source files
include("auxiliary/auxiliary.jl")
include("semidiscretization/semidiscretization.jl")
include("solvers/solvers.jl")

# Export the public APIs
export SemidiscretizationHyperbolicGPU
export semidiscretizeGPU

# Get called every time the package is loaded
function __init__()

    # Initialize the device properties
    init_device()
end

end
