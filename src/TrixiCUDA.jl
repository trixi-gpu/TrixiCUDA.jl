module TrixiCUDA

# Include other packages that are used in TrixiCUDA.jl
# using Reexport: @reexport

using CUDA
using CUDA: @cuda, CuArray, HostKernel,
            threadIdx, blockIdx, blockDim, reshape, similar, launch_configuration

# Trixi.jl methods
using Trixi: allocate_coefficients, compute_coefficients, create_cache,
             flux, ntuple, nvariables, ndofs,
             local_leaf_cells,
             init_elements, init_interfaces, init_boundaries, init_mortars,
             have_nonconservative_terms, boundary_condition_periodic,
             digest_boundary_conditions, check_periodicity_mesh_boundary_conditions,
             gauss_lobatto_nodes_weights, vandermonde_legendre,
             calc_dsplit, calc_dhat, calc_lhat,
             calc_forward_upper, calc_forward_lower, calc_reverse_upper, calc_reverse_lower,
             polynomial_derivative_matrix,
             set_log_type!, set_sqrt_type!,
             summary_header, summary_line, summary_footer, increment_indent # IO functions

# Trixi.jl structs
using Trixi: AbstractEquations, AbstractContainer, AbstractMesh, AbstractSemidiscretization,
             AbstractSurfaceIntegral, AbstractBasisSBP,
             PerformanceCounter, True, False, TreeMesh, StructuredMesh,
             DG, SemidiscretizationHyperbolic,
             LobattoLegendreBasis, LobattoLegendreMortarL2,
             ElementContainer1D, ElementContainer2D, ElementContainer3D,
             InterfaceContainer1D, InterfaceContainer2D, InterfaceContainer3D,
             BoundaryContainer1D, BoundaryContainer2D, BoundaryContainer3D,
             L2MortarContainer2D, L2MortarContainer3D,
             SurfaceIntegralWeakForm, VolumeIntegralWeakForm,
             BoundaryConditionPeriodic, UnstructuredSortedBoundaryTypes,
             VolumeIntegralFluxDifferencing, VolumeIntegralShockCapturingHG

# Trixi.jl indicators (since it is temporarily used to make tests pass,
# we separate it from the above)
using Trixi: IndicatorHennemannGassner,
             eachelement, calc_indicator_hennemann_gassner!, apply_smoothing!

# Trixi.jl imports
import Trixi: get_nodes, get_node_vars, get_node_coords, get_surface_node_vars,
              nnodes, nelements, ninterfaces, nmortars,
              eachnode, integrate,
              wrap_array, wrap_array_native, calc_error_norms,
              mesh_equations_solver_cache

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
export LobattoLegendreBasisGPU
export DGSEMGPU
export SemidiscretizationHyperbolicGPU
export semidiscretizeGPU

# Get called every time the package is loaded
function __init__()

    # Initialize the device properties
    init_device()
end

end
