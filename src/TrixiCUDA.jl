module TrixiCUDA

# GPU arrays and kernel launches
import CUDA
using CUDA: CuArray, CuDynamicSharedArray
using CUDA: @cuda, HostKernel, launch_configuration
using CUDA: threadIdx, blockIdx, blockDim, sync_threads
using GPUArraysCore: AbstractGPUArray, AbstractGPUVector

# Formatting, ODE problems, and static arrays
using Printf: @printf, @sprintf
using SciMLBase: ODEProblem, FullSpecialize
using StaticArrays: SVector

# Trixi abstractions, meshes, and solvers
using Trixi: AbstractEquations, AbstractContainer, AbstractMesh, AbstractSemidiscretization
using Trixi: TreeMesh, StructuredMesh, DG, DGSEM
using Trixi: True, False
using Trixi: PerformanceCounter

# Basis, mortar, and volume-integral types
using Trixi: AbstractBasisSBP, LobattoLegendreBasis, LobattoLegendreMortarL2, LobattoLegendreAnalyzer
using Trixi: AbstractSurfaceIntegral, SurfaceIntegralWeakForm
using Trixi: VolumeIntegralWeakForm, VolumeIntegralFluxDifferencing, VolumeIntegralShockCapturingHGType

# Tree-mesh containers
using Trixi: TreeElementContainer1D, TreeElementContainer2D, TreeElementContainer3D
using Trixi: TreeInterfaceContainer1D, TreeInterfaceContainer2D, TreeInterfaceContainer3D
using Trixi: TreeBoundaryContainer1D, TreeBoundaryContainer2D, TreeBoundaryContainer3D
using Trixi: TreeL2MortarContainer2D, TreeL2MortarContainer3D

# Equations, boundary conditions, and mesh initialization
using Trixi: flux, flux_central, have_nonconservative_terms, nvariables, max_abs_speeds
using Trixi: BoundaryConditionPeriodic, UnstructuredSortedBoundaryTypes, boundary_condition_periodic
using Trixi: digest_boundary_conditions, check_periodicity_mesh_boundary_conditions
using Trixi: local_leaf_cells, init_elements, init_interfaces, init_boundaries, init_mortars
using Trixi: allocate_coefficients, eachelement, ndofs, total_volume

# Interpolation, indicators, and analysis
using Trixi: gauss_lobatto_nodes_weights, polynomial_interpolation_matrix, multiply_dimensionwise!
using Trixi: calc_forward_upper, calc_forward_lower, calc_reverse_upper, calc_reverse_lower
using Trixi: IndicatorHennemannGassner, calc_indicator_hennemann_gassner!, apply_smoothing!
using Trixi: analyze, pretty_form_utf

# Preferences and display
using Trixi: set_log_type!, set_sqrt_type!, set_loop_vectorization!
using Trixi: summary_header, summary_line, summary_footer, increment_indent

# Trixi functions and constructors extended by this package
import Trixi: get_nodes, nnodes, polydeg, eachnode, SolutionAnalyzer
import Trixi: get_node_vars, get_node_coords, get_surface_node_vars
import Trixi: nelements, ninterfaces, nmortars
import Trixi: create_cache, mesh_equations_solver_cache, wrap_array, wrap_array_native
import Trixi: integrate, integrate_via_indices, calc_error_norms, analyze_integrals, max_dt

# Change to use the Base.log and Base.sqrt and disable loop vectorization
# FIXME: This set preference part has to be fixed to avoid precompilation outputs 
# ```[ Info: Please restart Julia and reload Trixi.jl for the `log` computation change to take effect
#    [ Info: Please restart Julia and reload Trixi.jl for the `sqrt` computation change to take effect
#    [ Info: Please restart Julia and reload Trixi.jl for the `loop_vectorization` change to take effect```
set_log_type!("log_Base")
set_sqrt_type!("sqrt_Base")
set_loop_vectorization!(false)

# Include other source files
include("auxiliary/auxiliary.jl")
include("semidiscretization/semidiscretization.jl")
include("solvers/solvers.jl")
include("callbacks_step/callbacks_step.jl")

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
