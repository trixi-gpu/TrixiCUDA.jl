module TrixiCUDA

# Include other packages that are used in TrixiCUDA.jl
using CUDA
using CUDA: @cuda, CuArray, HostKernel,
            threadIdx, blockIdx, blockDim, reshape, similar, launch_configuration

using Printf: @printf, @sprintf
using SciMLBase: ODEProblem, FullSpecialize
using StaticArrays: SVector

# Trixi.jl methods
using Trixi: allocate_coefficients, compute_coefficients, create_cache,
             flux, ntuple, nvariables, ndofs,
             local_leaf_cells,
             init_elements, init_interfaces, init_boundaries, init_mortars,
             have_nonconservative_terms, boundary_condition_periodic,
             digest_boundary_conditions, check_periodicity_mesh_boundary_conditions,
             gauss_lobatto_nodes_weights, vandermonde_legendre,
             polynomial_interpolation_matrix, volume_jacobian, total_volume,
             calc_dsplit, calc_dhat, calc_lhat,
             calc_forward_upper, calc_forward_lower, calc_reverse_upper, calc_reverse_lower,
             polynomial_derivative_matrix,
             analyze, cons2cons, max_abs_speeds, pretty_form_utf,
             multiply_dimensionwise!,
             set_log_type!, set_sqrt_type!, set_loop_vectorization!,
             summary_header, summary_line, summary_footer, increment_indent # IO functions

# Trixi.jl structs
using Trixi: AbstractEquations, AbstractContainer, AbstractMesh, AbstractSemidiscretization,
             AbstractSurfaceIntegral, AbstractBasisSBP,
             PerformanceCounter, True, False, TreeMesh, StructuredMesh,
             DG, FDSBP, SemidiscretizationHyperbolic,
             LobattoLegendreBasis, LobattoLegendreMortarL2,
             ElementContainer1D, ElementContainer2D, ElementContainer3D,
             InterfaceContainer1D, InterfaceContainer2D, InterfaceContainer3D,
             BoundaryContainer1D, BoundaryContainer2D, BoundaryContainer3D,
             L2MortarContainer2D, L2MortarContainer3D,
             SurfaceIntegralWeakForm, VolumeIntegralWeakForm,
             BoundaryConditionPeriodic, UnstructuredSortedBoundaryTypes,
             VolumeIntegralFluxDifferencing, VolumeIntegralShockCapturingHG,
             LobattoLegendreAnalyzer

# Trixi.jl indicators (since it is temporarily used to make tests pass,
# we separate it from the above)
using Trixi: IndicatorHennemannGassner,
             eachelement, calc_indicator_hennemann_gassner!, apply_smoothing!

# Trixi.jl imports
import Trixi: get_nodes, get_node_vars, get_node_coords, get_surface_node_vars,
              nnodes, nelements, ninterfaces, nmortars,
              polydeg, eachnode, integrate,
              wrap_array, wrap_array_native, calc_error_norms,
              create_cache, mesh_equations_solver_cache,
              integrate_via_indices, analyze_integrals, max_dt,
              SolutionAnalyzer # function

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
