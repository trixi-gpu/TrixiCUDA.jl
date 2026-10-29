module TrixiCUDA

# GPU arrays and kernel launches
import CUDA
using CUDA: CuArray, CuDynamicSharedArray, CUDABackend,
            @cuda, HostKernel, launch_configuration,
            threadIdx, blockIdx, blockDim, sync_threads
using GPUArraysCore: AbstractGPUArray

# Formatting, ODE problems, and static arrays
using Printf: @printf, @sprintf
using SciMLBase: ODEProblem, ODEFunction, FullSpecialize
using StaticArrays: SVector

# Trixi abstractions, meshes, and solvers
using Trixi: AbstractEquations, AbstractContainer, AbstractMesh, AbstractSemidiscretization,
             TreeMesh, TreeMeshSerial, StructuredMesh, DGSEM,
             True, False, PerformanceCounter

# Basis, mortar, and volume-integral types
using Trixi: AbstractBasisSBP, LobattoLegendreBasis, LobattoLegendreMortarL2, LobattoLegendreAnalyzer,
             AbstractSurfaceIntegral, SurfaceIntegralWeakForm,
             VolumeIntegralWeakForm, VolumeIntegralFluxDifferencing, VolumeIntegralShockCapturingHGType,
             VolumeIntegralPureLGLFiniteVolume

# Tree-mesh containers
using Trixi: TreeElementContainer1D, TreeElementContainer2D, TreeElementContainer3D,
             TreeInterfaceContainer1D, TreeInterfaceContainer2D, TreeInterfaceContainer3D,
             TreeBoundaryContainer1D, TreeBoundaryContainer2D, TreeBoundaryContainer3D,
             TreeL2MortarContainer2D, TreeL2MortarContainer3D

# Equations, boundary conditions, and mesh initialization
using Trixi: flux, flux_central, have_nonconservative_terms, nvariables, max_abs_speeds,
             BoundaryConditionPeriodic, UnstructuredSortedBoundaryTypes, boundary_condition_periodic,
             digest_boundary_conditions, check_periodicity_mesh_boundary_conditions,
             local_leaf_cells, init_elements, init_interfaces, init_boundaries, init_mortars,
             allocate_coefficients, eachelement, ndofs, total_volume

# Interpolation, indicators, and analysis
using Trixi: gauss_lobatto_nodes_weights, polynomial_interpolation_matrix, multiply_dimensionwise!,
             calc_forward_upper, calc_forward_lower, calc_reverse_upper, calc_reverse_lower,
             IndicatorHennemannGassner, calc_indicator_hennemann_gassner!, apply_smoothing!,
             analyze, pretty_form_utf

# Backend selection and time conversion shared with upstream Trixi
using Trixi: trixi_backend, convert_tspan

# Display
using Trixi: summary_header, summary_line, summary_footer, increment_indent

# Trixi functions and constructors extended by this package
import Trixi: get_nodes, nnodes, polydeg, eachnode, SolutionAnalyzer,
              get_node_vars, get_node_coords, get_surface_node_vars,
              nelements, ninterfaces, nmortars,
              create_cache, mesh_equations_solver_cache, wrap_array, wrap_array_native,
              integrate, integrate_via_indices, calc_error_norms, analyze_integrals, max_dt

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
