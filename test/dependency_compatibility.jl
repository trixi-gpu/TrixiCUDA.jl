using Test
using CUDA, Trixi, TrixiCUDA

@testset "Dependency compatibility" begin
    equations = LinearScalarAdvectionEquation1D(1.0)
    basis = LobattoLegendreBasis(3)
    indicator = IndicatorHennemannGassner(equations, basis; variable = (u, equations) -> first(u))
    volume_integral = VolumeIntegralShockCapturingHG(indicator)
    @test TrixiCUDA.check_gpu_volume_integral(volume_integral) === nothing
    @test TrixiCUDA.check_gpu_volume_integral(VolumeIntegralWeakForm()) === nothing

    generalized = VolumeIntegralShockCapturingHGType(indicator;
                                                     volume_integral_default = VolumeIntegralWeakForm(),
                                                     volume_integral_blend_high_order = VolumeIntegralFluxDifferencing(flux_central),
                                                     volume_integral_blend_low_order = VolumeIntegralPureLGLFiniteVolume(flux_lax_friedrichs))
    @test_throws ArgumentError TrixiCUDA.check_gpu_volume_integral(generalized)
    different_fluxes = VolumeIntegralShockCapturingHGType(indicator;
                                                          volume_integral_default = VolumeIntegralFluxDifferencing(flux_central),
                                                          volume_integral_blend_high_order = VolumeIntegralFluxDifferencing(flux_lax_friedrichs),
                                                          volume_integral_blend_low_order = VolumeIntegralPureLGLFiniteVolume(flux_lax_friedrichs))
    @test_throws ArgumentError TrixiCUDA.check_gpu_volume_integral(different_fluxes)

    # These public CUDA constants must resolve even on hosts without a GPU.
    @test isdefined(CUDA, :DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    @test isdefined(CUDA, :DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    @test isdefined(CUDA, :DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)

    # Exercise the host reference interfaces used by the GPU comparison tests.
    for spatial_dimension in 1:3
        equations = getproperty(Trixi, Symbol("LinearScalarAdvectionEquation$(spatial_dimension)D"))(spatial_dimension == 1 ? 1.0 :
                                                                                                     ntuple(_ -> 1.0, spatial_dimension))
        coordinates_min = spatial_dimension == 1 ? -1.0 : ntuple(_ -> -1.0, spatial_dimension)
        coordinates_max = spatial_dimension == 1 ? 1.0 : ntuple(_ -> 1.0, spatial_dimension)
        mesh = TreeMesh(coordinates_min, coordinates_max;
                        initial_refinement_level = 1, periodicity = false)
        solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                            solver;
                                            boundary_conditions = BoundaryConditionDirichlet(initial_condition_convergence_test))
        ode = semidiscretize(semi, (0.0, 0.1))
        (; cache) = semi
        u = Trixi.wrap_array(ode.u0, mesh, equations, solver, cache)
        du = similar(u)
        Trixi.set_zero!(du, solver, cache)
        Trixi.calc_volume_integral!(nothing, du, u, mesh,
                                    Trixi.have_nonconservative_terms(equations), equations,
                                    solver.volume_integral, solver, cache)
        if spatial_dimension == 1
            Trixi.prolong2interfaces!(cache, u, mesh, equations, solver)
            Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                                       Trixi.have_nonconservative_terms(equations), equations,
                                       solver.surface_integral, solver, cache)
        else
            Trixi.prolong2interfaces!(nothing, cache, u, mesh, equations, solver)
            Trixi.calc_interface_flux!(nothing, cache.elements.surface_flux_values, mesh,
                                       Trixi.have_nonconservative_terms(equations), equations,
                                       solver.surface_integral, solver, cache)
        end
        Trixi.prolong2boundaries!(nothing, cache, u, mesh, equations, solver)
        Trixi.calc_boundary_flux!(nothing, cache, 0.0, semi.boundary_conditions,
                                  mesh, equations, solver.surface_integral, solver)
        Trixi.calc_surface_integral!(nothing, du, u, mesh, equations, solver.surface_integral,
                                     solver, cache)
        Trixi.apply_jacobian!(nothing, du, mesh, equations, solver, cache)
        Trixi.calc_sources!(nothing, du, u, 0.0, nothing, equations, solver, cache)
        reference = similar(ode.u0)
        ode.f(reference, ode.u0, ode.p, 0.0)
        @test vec(du) ≈ reference
        @test all(isfinite, du)

        # Check GPU-vector dispatch without allocating device memory. This also
        # runs in the CPU-only dependency CI jobs.
        for RealT in (Float32, Float64)
            argument_types = Tuple{CuArray{RealT, 1, CUDA.DeviceMemory}, typeof(mesh),
                                   typeof(equations), typeof(solver), typeof(cache)}
            @test which(Trixi.wrap_array, argument_types).module === TrixiCUDA
            @test which(Trixi.wrap_array_native, argument_types).module === TrixiCUDA
        end

        if CUDA.functional()
            solver_gpu = DGSEMGPU(polydeg = 3, surface_flux = flux_lax_friedrichs)
            u_gpu_flat = CuArray(ode.u0)
            GC.@preserve u_gpu_flat begin
                for wrap in (Trixi.wrap_array, Trixi.wrap_array_native)
                    u_gpu = wrap(u_gpu_flat, mesh, equations, solver_gpu, cache)
                    @test size(u_gpu) == size(u)
                    @test Array(u_gpu) ≈ Array(u)
                    @test pointer(u_gpu) == pointer(u_gpu_flat)
                end
            end
        end
    end

    if CUDA.functional()
        @test TrixiCUDA.MULTIPROCESSOR_COUNT > 0
        @test TrixiCUDA.MAX_THREADS_PER_BLOCK > 0
        @test TrixiCUDA.MAX_SHARED_MEMORY_PER_BLOCK > 0
        for RealT in (Float32, Float64), degree in (1, 3, 5)
            basis_cpu = LobattoLegendreBasis(RealT, degree)
            basis_gpu = LobattoLegendreBasisGPU(RealT, degree)
            @test Array(basis_gpu.derivative_hat) ≈ basis_cpu.derivative_hat
            @test Array(basis_gpu.derivative_split) ≈ basis_cpu.derivative_split
            @test Array(basis_gpu.derivative_split_transpose) ≈ transpose(basis_cpu.derivative_split)
            expected_boundary = zeros(RealT, degree + 1, 2)
            expected_boundary[1, 1] = first(basis_cpu.inverse_weights)
            expected_boundary[end, 2] = last(basis_cpu.inverse_weights)
            @test basis_gpu.boundary_interpolation ≈ expected_boundary
            indicator = IndicatorHennemannGassner(LinearScalarAdvectionEquation1D(one(RealT)), basis_gpu;
                                                  variable = (u, equations) -> first(u))
            @test length(indicator.cache.indicator_threaded) == Threads.maxthreadid()
        end
    end
end
