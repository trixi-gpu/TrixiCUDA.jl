#include("header.jl") # Remove it after first run to avoid recompilation

# Set random seed
Random.seed!(12345)

# The header part of test
equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test

# Note that the expected EOC of 5 is not reached with this flux.
# Using flux_hll instead yields the expected EOC.
solver = DGSEM(polydeg=4, surface_flux=flux_lax_friedrichs)

coordinates_min = 0.0
coordinates_max = 2.0
mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=4,
    n_cells_max=10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
    source_terms=source_terms_convergence_test)

@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

t = 0.0
l = nvariables(equations) * nnodes(solver)^ndims(mesh) * nelements(solver, cache)
u_ode = rand(Float64, l)
du_ode = rand(Float64, l)
u = wrap_array(u_ode, mesh, equations, solver, cache)
du = wrap_array(du_ode, mesh, equations, solver, cache)

# Source terms kernel for 1D equations except for `CompressibleEulerMulticomponentEquations1D`
# i.e. `CompressibleEulerEquations1D`, `InviscidBurgersEquation1D`, `ShallowWaterEquations1D`, 
# `ShallowWaterEquations1D`, and `ShallowWaterTwoLayerEquations1D`
#################################################################################
function source_terms_kernel!(du, u, node_coordinates, t, equations::AbstractEquations{1}, source_terms::Function) # use union
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        @inbounds du[i, j, k] += source_terms(u[i, j, k], node_coordinates[1, j, k], t, equations)[i]
    end

    return nothing
end

function cuda_source_terms!(du, u, t, source_terms,
    equations::AbstractEquations{1}, dg::DG, cache)

    node_coordinates = CuArray{Float32}(cache.elements.node_coordinates)

    source_terms_kernel = @cuda launch = false source_terms_kernel!(du, u, node_coordinates, t, equations, source_terms)
    source_terms_kernel(du, u, node_coordinates, t, equations, source_terms; configurator_3d(source_terms_kernel, du)...)

    return nothing
end

# Source terms kernel for all 1D equations 
# i.e. includes `CompressibleEulerMulticomponentEquations1D`
#################################################################################
function source_terms_kernel!(du, u, node_coordinates, t, equations::AbstractEquations{1}, source_terms::Function)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        u_local = []
        for ii in 1:size(du, 1)
            push!(u_local, du[ii, j, k])
        end
        @inbounds du[i, j, k] += source_terms(u_local, node_coordinates[1, j, k], t, equations)[i]
    end

    return nothing
end

function cuda_source_terms!(du, u, t, source_terms,
    equations::AbstractEquations{1}, dg::DG, cache)

    node_coordinates = CuArray{Float32}(cache.elements.node_coordinates)

    source_terms_kernel = @cuda launch = false source_terms_kernel!(du, u, node_coordinates, t, equations, source_terms)
    source_terms_kernel(du, u, node_coordinates, t, equations, source_terms; configurator_3d(source_terms_kernel, du)...)

    return nothing
end


