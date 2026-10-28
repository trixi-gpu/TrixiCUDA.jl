using Trixi, TrixiShallowWater
using StaticArrays: SVector
using StableRNGs: StableRNG
using Random: rand

# Match every existing CPU/GPU comparison script in benchmark/, using the same
# meshes, degrees, fluxes, boundary conditions, and source terms. Only u changes.
function rhs_cases()
    cases = NamedTuple[]
    for (prefix, family, variant, dimensions) in (("advection_basic", :advection, :basic, 1:3),
        ("advection_mortar", :advection, :mortar, 2:3),
        ("euler_ec", :euler, :ec, 1:3),
        ("euler_shock", :euler, :shock, 1:3),
        ("hypdiff_nonperiodic", :hypdiff, :nonperiodic, 1:3),
        ("mhd_alfven_wave", :mhd, :alfven, 1:3),
        ("mhd_alfven_wave_mortar", :mhd, :mortar, 2:3),
        ("mhd_ec", :mhd, :ec, 1:3),
        ("shallowwater_ec", :shallowwater, :ec, 1:2),
        ("shallowwater_source_terms", :shallowwater, :source, 1:2))
        for dim in dimensions
            push!(cases, (; name = "$(prefix)_$(dim)d", family, variant, dim))
        end
    end
    return sort!(cases; by = c -> c.name)
end

function make_equations(family, variant, dim)
    if family == :advection
        constructors = (LinearScalarAdvectionEquation1D, LinearScalarAdvectionEquation2D,
                        LinearScalarAdvectionEquation3D)
        velocities = (1.0f0, (0.2f0, -0.7f0), (0.2f0, -0.7f0, 0.5f0))
        return constructors[dim](velocities[dim])
    elseif family == :euler
        return (CompressibleEulerEquations1D, CompressibleEulerEquations2D,
                CompressibleEulerEquations3D)[dim](1.4f0)
    elseif family == :hypdiff
        return (HyperbolicDiffusionEquations1D, HyperbolicDiffusionEquations2D,
                HyperbolicDiffusionEquations3D)[dim](nu = 1.0f0, Lr = inv(2 * Float32(pi)))
    elseif family == :mhd
        gamma = variant == :ec ? (dim == 1 ? 2.0f0 : 1.4f0) : 5.0f0 / 3
        if dim == 1
            return IdealGlmMhdEquations1D(gamma)
        end
        return (IdealGlmMhdEquations2D, IdealGlmMhdEquations3D)[dim - 1](gamma;
                                                                         initial_c_h = 1.0f0)
    else
        # TrixiShallowWater requires explicit non-Float64 thresholds. All random
        # inputs below are fully wet; these match its Float64 default values.
        return (ShallowWaterEquations1D, ShallowWaterEquations2D)[dim](gravity = 9.81f0,
                                                                       threshold_partially_wet = 1.0f-4,
                                                                       threshold_desingularization = 1.0f-10)
    end
end

function make_solver(case, equations, solver_type)
    (; family, variant, dim) = case
    polydeg = family == :hypdiff || (family == :shallowwater && variant == :ec) ? 4 : 3
    surface_flux = flux_lax_friedrichs
    volume_integral = VolumeIntegralWeakForm()
    if family == :euler
        if variant == :ec
            surface_flux = flux_ranocha
            volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha)
        else
            surface_flux = dim == 3 ? flux_ranocha : flux_lax_friedrichs
            volume_flux = dim == 3 ? flux_ranocha : flux_shima_etal
            basis = LobattoLegendreBasis(Float32, polydeg)
            indicator = IndicatorHennemannGassner(equations, basis;
                                                  alpha_max = 0.5f0, alpha_min = 0.001f0,
                                                  alpha_smooth = true, variable = density_pressure)
            volume_integral = VolumeIntegralShockCapturingHG(indicator;
                                                             volume_flux_dg = volume_flux,
                                                             volume_flux_fv = surface_flux)
        end
    elseif family == :mhd
        if dim == 1
            surface_flux = variant == :ec ? flux_hindenlang_gassner : flux_lax_friedrichs
            volume_flux = flux_hindenlang_gassner
        else
            conservative_volume_flux = variant == :alfven && dim == 2 ? flux_central : flux_hindenlang_gassner
            conservative_surface_flux = variant == :ec ? flux_hindenlang_gassner :
                                        variant == :mortar ? flux_hlle : flux_lax_friedrichs
            surface_flux = (conservative_surface_flux, flux_nonconservative_powell)
            volume_flux = (conservative_volume_flux, flux_nonconservative_powell)
        end
        volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
    elseif family == :shallowwater
        conservative_surface_flux = variant == :ec ? flux_fjordholm_etal : flux_lax_friedrichs
        surface_flux = (conservative_surface_flux, flux_nonconservative_fjordholm_etal)
        volume_integral = VolumeIntegralFluxDifferencing((flux_wintermeyer_etal,
                                                          flux_nonconservative_wintermeyer_etal))
    end
    return solver_type(; polydeg, surface_flux, volume_integral, RealT = Float32)
end

function make_mesh(case)
    (; family, variant, dim) = case
    lower, upper = -1.0f0, 1.0f0
    level = family == :mhd || variant == :mortar ||
            (family == :shallowwater && variant == :ec) ? 2 : 3
    if family == :euler || (family == :mhd && variant == :ec && dim > 1)
        lower, upper = -2.0f0, 2.0f0
    elseif family == :hypdiff || (family == :mhd && dim == 1)
        lower, upper = 0.0f0, 1.0f0
    elseif (family == :mhd && dim == 2) || (family == :shallowwater && variant == :source)
        lower, upper = 0.0f0, sqrt(2.0f0)
    end
    if family == :shallowwater && variant == :source
        level = 4
    end
    coordinates_min, coordinates_max = ntuple(_ -> lower, dim), ntuple(_ -> upper, dim)
    patches = ()
    if variant == :mortar
        if family == :advection
            patches = ((type = "box", coordinates_min = ntuple(i -> i == 1 ? 0.0f0 : -1.0f0, dim),
                        coordinates_max),)
            if dim == 3
                patches = (patches...,
                           (type = "box", coordinates_min = (0.0f0, -0.5f0, -0.5f0),
                            coordinates_max = (0.5f0, 0.5f0, 0.5f0)))
            end
        else
            patches = ((type = "box", coordinates_min = dim == 2 ? 0.25f0 .* coordinates_max : (-0.5f0, -0.5f0, -0.5f0),
                        coordinates_max = dim == 2 ? 0.75f0 .* coordinates_max : (0.5f0, 0.5f0, 0.5f0)),)
        end
    end
    periodicity = family == :hypdiff ? ntuple(i -> i != 1, dim) : true
    return TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level = level, refinement_patches = patches,
                    periodicity, RealT = Float32)
end

# Constructors are arguments so the exact same setups can also be checked on CPU.
function make_problem(case, solver_type, semi_type)
    equations = make_equations(case.family, case.variant, case.dim)
    solver = make_solver(case, equations, solver_type)
    mesh = make_mesh(case)
    boundary_conditions = boundary_condition_periodic
    source_terms = nothing
    if case.family == :hypdiff
        boundary_names = (:x_neg, :x_pos, :y_neg, :y_pos, :z_neg, :z_pos)[1:(2 * case.dim)]
        boundary_values = ntuple(i -> i <= 2 ? boundary_condition_poisson_nonperiodic : boundary_condition_periodic, 2 * case.dim)
        boundary_conditions = NamedTuple{boundary_names}(boundary_values)
        source_terms = source_terms_poisson_nonperiodic
    elseif (case.family == :shallowwater && case.variant == :source) ||
           (case.family == :euler && case.variant == :shock && case.dim == 3)
        source_terms = source_terms_convergence_test
    end
    # No initial condition or ODE solve is evaluated; u is supplied directly below.
    return semi_type(mesh, equations, nothing, solver; boundary_conditions, source_terms)
end

function random_state(rng, equations, family)
    values = rand(rng, Float32, Trixi.nvariables(equations))
    if family == :euler || family == :mhd
        # Random primitive variables avoid negative density/pressure in random
        # conservative vectors. MHD always has three velocity components.
        values[1] += 1.0f0
        values[family == :euler ? ndims(equations) + 2 : 5] += 1.0f0
        return prim2cons(SVector{length(values)}(values), equations)
    elseif family == :shallowwater
        # Shallow-water primitives use total height H = h + bottom topography.
        values[1] += 1.0f0 + values[end]
        return prim2cons(SVector{length(values)}(values), equations)
    end
    return SVector{length(values)}(values)
end

function random_input(case, equations, solver, cache, seed)
    nv = Trixi.nvariables(equations)
    nn = Trixi.nnodes(solver)
    ne = Trixi.nelements(solver, cache)
    u = Array{Float32}(undef, nv, ntuple(_ -> nn, case.dim)..., ne)
    nodes = reshape(u, nv, :)
    rng = StableRNG(seed)
    for j in axes(nodes, 2)
        nodes[:, j] = random_state(rng, equations, case.family)
    end
    return u
end
