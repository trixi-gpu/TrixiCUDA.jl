module GeneralizedHG

using Trixi, TrixiCUDA, Test
import CUDA

# Setups below are copied from Trixi v0.17.19 (56de0a2fe863343cd1a275415a966b93449ece55).
# Keep the equations, initial/boundary data, and HG/RRG settings from upstream.
# Only reduce refinement for RHS tests; omit time integration and AMR callbacks.

# Source: https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_1d_dgsem/elixir_euler_shu_osher.jl
function setup_shu_osher()
    # semidiscretization of the compressible Euler equations

    equations = CompressibleEulerEquations1D(1.4)

    # Shu-Osher initial condition for 1D compressible Euler equations
    # Example 8 from Shu, Osher (1989).
    # https://doi.org/10.1016/0021-9991(89)90222-2
    function initial_condition_shu_osher(x, t, equations::CompressibleEulerEquations1D)
        x0 = -4

        rho_left = 27 / 7
        v_left = 4 * sqrt(35) / 9
        p_left = 31 / 3

        v_right = 0.0
        p_right = 1.0

        rho = ifelse(x[1] > x0, 1 + 1 / 5 * sin(5 * x[1]), rho_left)
        v = ifelse(x[1] > x0, v_right, v_left)
        p = ifelse(x[1] > x0, p_right, p_left)

        return prim2cons(SVector(rho, v, p), equations)
    end

    initial_condition = initial_condition_shu_osher

    surface_flux = flux_hllc
    volume_flux = flux_ranocha
    polydeg = 4
    basis = LobattoLegendreBasis(polydeg)
    shock_indicator_variable = density_pressure
    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max = 1.0,
                                             alpha_min = 0.001,
                                             alpha_smooth = true,
                                             variable = shock_indicator_variable)
    volume_integral = VolumeIntegralShockCapturingRRG(basis, indicator_sc;
                                                      volume_flux_dg = volume_flux,
                                                      volume_flux_fv = surface_flux,
                                                      slope_limiter = monotonized_central)

    solver = DGSEM(basis, surface_flux, volume_integral)

    coordinates_min = (-5.0,)
    coordinates_max = (5.0,)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 6,
                    periodicity = false)

    boundary_conditions = BoundaryConditionDirichlet(initial_condition)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions = boundary_conditions)
    return semi
end

# Source: https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_2d_dgsem/elixir_euler_astro_jet_amr_scO2.jl
function setup_astro_jet()
    # semidiscretization of the compressible Euler equations
    gamma = 5 / 3
    equations = CompressibleEulerEquations2D(gamma)

    # Initial condition adopted from
    # - Yong Liu, Jianfang Lu, and Chi-Wang Shu
    #   An oscillation free discontinuous Galerkin method for hyperbolic systems
    #   https://doi.org/10.1137/20M1354192
    # Mach = 2000 jet
    function initial_condition_astro_jet(x, t, equations::CompressibleEulerEquations2D)
        RealT = eltype(x)
        @unpack gamma = equations
        rho = 0.5f0
        v1 = 0
        v2 = 0
        p = convert(RealT, 0.4127)
        # add inflow for t>0 at x=-0.5
        # domain size is [-0.5,+0.5]^2
        if (t > 0) && (x[1] ≈ -0.5f0) && (abs(x[2]) < RealT(0.05))
            rho = 5.0f0
            v1 = 800 # about Mach number Ma = 2000
            v2 = 0
            p = convert(RealT, 0.4127)
        end
        return prim2cons(SVector(rho, v1, v2, p), equations)
    end
    initial_condition = initial_condition_astro_jet

    boundary_conditions = (; x_neg = BoundaryConditionDirichlet(initial_condition_astro_jet),
                           x_pos = BoundaryConditionDirichlet(initial_condition_astro_jet),
                           y_neg = boundary_condition_periodic,
                           y_pos = boundary_condition_periodic)

    surface_flux = flux_hll
    volume_flux = flux_ranocha
    polydeg = 3
    basis = LobattoLegendreBasis(polydeg)

    # shock capturing necessary for this tough example
    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max = 0.3,
                                             alpha_min = 0.0001,
                                             alpha_smooth = true,
                                             variable = density_pressure)
    volume_integral = VolumeIntegralShockCapturingRRG(basis, indicator_sc;
                                                      volume_flux_dg = volume_flux,
                                                      volume_flux_fv = surface_flux,
                                                      slope_limiter = minmod)

    solver = DGSEM(basis, surface_flux, volume_integral)

    coordinates_min = (-0.5, -0.5)
    coordinates_max = (0.5, 0.5)

    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 3,
                    periodicity = (false, true))
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions = boundary_conditions)
    return semi
end

# Source: https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_euler_sedov_scO2.jl
function setup_sedov_rrg()
    # semidiscretization of the compressible Euler equations

    equations = CompressibleEulerEquations3D(1.4)

    """
        initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations3D)

    The Sedov blast wave setup based on example 35.1.4 from Flash
    - https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
    """
    function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations3D)
        # Set up polar coordinates
        RealT = eltype(x)
        inicenter = SVector(0, 0, 0)
        x_norm = x[1] - inicenter[1]
        y_norm = x[2] - inicenter[2]
        z_norm = x[3] - inicenter[3]
        r = sqrt(x_norm^2 + y_norm^2 + z_norm^2)

        # Setup based on example 35.1.4 in https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
        r0 = 0.21875f0 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
        E = 1
        nu = 3 # dims
        p0_inner = 3 * (equations.gamma - 1) * E / ((nu + 1) * convert(RealT, pi) * r0^nu)
        p0_outer = convert(RealT, 1.0e-5) # = true Sedov setup

        # Calculate primitive variables
        rho = 1
        v1 = 0
        v2 = 0
        v3 = 0
        p = r > r0 ? p0_outer : p0_inner

        return prim2cons(SVector(rho, v1, v2, v3, p), equations)
    end
    initial_condition = initial_condition_sedov_blast_wave

    surface_flux = flux_lax_friedrichs
    volume_flux = flux_chandrashekar
    basis = LobattoLegendreBasis(3)
    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max = 0.5,
                                             alpha_min = 0.001,
                                             alpha_smooth = true,
                                             variable = density_pressure)
    volume_integral = VolumeIntegralShockCapturingRRG(basis, indicator_sc;
                                                      volume_flux_dg = volume_flux,
                                                      volume_flux_fv = surface_flux,
                                                      slope_limiter = minmod)

    solver = DGSEM(basis, surface_flux, volume_integral)

    coordinates_min = (-2.0, -2.0, -2.0)
    coordinates_max = (2.0, 2.0, 2.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 3,
                    periodicity = true)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions = boundary_condition_periodic)
    return semi
end

# Source: https://github.com/trixi-framework/Trixi.jl/blob/v0.17.19/examples/tree_3d_dgsem/elixir_euler_sedov_blast_weak_form_sc.jl
function setup_sedov_weak()
    # semidiscretization of the compressible Euler equations

    equations = CompressibleEulerEquations3D(1.4)

    """
        initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations2D)

    The Sedov blast wave setup based on example 35.1.4 from Flash
    - https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
    """
    function initial_condition_sedov_blast_wave(x, t, equations::CompressibleEulerEquations3D)
        # Set up polar coordinates
        RealT = eltype(x)
        inicenter = SVector(0, 0, 0)
        x_norm = x[1] - inicenter[1]
        y_norm = x[2] - inicenter[2]
        z_norm = x[3] - inicenter[3]
        r = sqrt(x_norm^2 + y_norm^2 + z_norm^2)

        # Setup based on example 35.1.4 in https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p8.pdf
        r0 = 0.21875f0 # = 3.5 * smallest dx (for domain length=4 and max-ref=6)
        E = 1
        nu = 3 # dims
        p0_inner = 3 * (equations.gamma - 1) * E / ((nu + 1) * convert(RealT, pi) * r0^nu)
        p0_outer = convert(RealT, 1.0e-5) # = true Sedov setup

        # Calculate primitive variables
        rho = 1
        v1 = 0
        v2 = 0
        v3 = 0
        p = r > r0 ? p0_outer : p0_inner

        return prim2cons(SVector(rho, v1, v2, v3, p), equations)
    end
    initial_condition = initial_condition_sedov_blast_wave

    basis = LobattoLegendreBasis(3)

    # Use standard Hennemann-Gassner a-priori shock & blending indicator
    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max = 0.5,
                                             alpha_min = 0.001,
                                             alpha_smooth = true,
                                             variable = density_pressure)

    # In non-blended/limited regions, we use the cheaper weak form volume integral
    volume_integral_default = VolumeIntegralWeakForm()

    surface_flux = flux_lax_friedrichs
    volume_flux = flux_chandrashekar
    # For the blended/limited regions, we need to supply high-order and low-order volume integrals.
    volume_integral_blend_high_order = VolumeIntegralFluxDifferencing(volume_flux)
    volume_integral_blend_low_order = VolumeIntegralPureLGLFiniteVolume(surface_flux)

    volume_integral = VolumeIntegralShockCapturingHGType(indicator_sc;
                                                         volume_integral_default = volume_integral_default,
                                                         volume_integral_blend_high_order = volume_integral_blend_high_order,
                                                         volume_integral_blend_low_order = volume_integral_blend_low_order)

    solver = DGSEM(basis, surface_flux, volume_integral)

    coordinates_min = (-2.0, -2.0, -2.0)
    coordinates_max = (2.0, 2.0, 2.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level = 3,
                    periodicity = true)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions = boundary_condition_periodic)
    return semi
end

@testset "Generalized HG configurations from Trixi" begin
    for (name, setup, t) in (("1D Shu-Osher RRG / MC", setup_shu_osher, 0.0),
        # At t>0 the upstream jet inflow is active, so the test
        # exercises reconstruction rather than only a constant state.
        ("2D astro jet RRG / minmod", setup_astro_jet, 1.0e-4),
        ("3D Sedov RRG / minmod", setup_sedov_rrg, 0.0),
        ("3D Sedov weak default / FD blend", setup_sedov_weak, 0.0))
        @testset "$name" begin
            semi = setup()
            (; mesh, equations, initial_condition, boundary_conditions, solver, cache) = semi
            solver_gpu = DGSEMGPU(polydeg = polydeg(solver),
                                  surface_integral = solver.surface_integral,
                                  volume_integral = solver.volume_integral)
            semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver_gpu;
                                                       boundary_conditions)
            u_ode = Trixi.compute_coefficients(t, semi)
            # No AMR here: reshape retains the parent storage across garbage collection.
            u = reshape(u_ode, Trixi.nvariables(equations),
                        ntuple(_ -> Trixi.nnodes(solver), ndims(mesh))..., Trixi.nelements(solver, cache))
            du_ode = zeros(eltype(u_ode), length(u_ode))
            du = reshape(du_ode, size(u))
            u_gpu_ode = CUDA.CuArray(u_ode)
            du_gpu_ode = similar(u_gpu_ode)
            u_gpu = TrixiCUDA.wrap_array(u_gpu_ode, mesh, equations, solver_gpu, semi_gpu.cache_gpu)
            du_gpu = TrixiCUDA.wrap_array(du_gpu_ode, mesh, equations, solver_gpu, semi_gpu.cache_gpu)

            alpha = solver.volume_integral.indicator(u, mesh, equations, solver, cache)
            @test any(>(0), alpha)
            if setup === setup_sedov_weak
                @test any(iszero, alpha) # Exercise the distinct default operator as well.
            end
            Trixi.calc_volume_integral!(nothing, du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                        equations, solver.volume_integral, solver, cache)
            TrixiCUDA.calc_volume_integral!(du_gpu, u_gpu, mesh, equations,
                                            solver_gpu.volume_integral, solver_gpu,
                                            semi_gpu.cache_gpu, semi_gpu.cache_cpu)
            @test all(isfinite, Array(du_gpu))
            @test Array(du_gpu) ≈ du

            Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, t)
            TrixiCUDA.rhs_hyperbolic!(du_gpu_ode, u_gpu_ode, semi_gpu, t)
            @test all(isfinite, Array(du_gpu_ode))
            @test Array(du_gpu_ode) ≈ du_ode
        end
        GC.gc(true)
        CUDA.reclaim()
    end
end

end # module
