module TestShallowWaterFluxDifferencing

####################################################################### Tags
# Kernels: 
#   -`cuda_volume_integral!`
# Conditions:
#   - `nonconservative_terms::True`
#   - `volume_integral::VolumeIntegralFluxDifferencing`
#######################################################################

include("test_trixicuda.jl")

# Test precision of the semidiscretization process
@testset "Test Shallow Water" begin
    @testset "Shallow Water 1D" begin
        equations = ShallowWaterEquations1D(gravity_constant = 9.81)

        function initial_condition_ec_discontinuous_bottom(x, t, equations::ShallowWaterEquations1D)
            # Set the background values
            H = 4.25
            v = 0.0
            b = sin(x[1]) # arbitrary continuous function

            # Setup the discontinuous water height and velocity
            if x[1] >= 0.125 && x[1] <= 0.25
                H = 5.0
                v = 0.1882
            end

            # Setup a discontinuous bottom topography
            if x[1] >= -0.25 && x[1] <= -0.125
                b = 2.0 + 0.5 * sin(2.0 * pi * x[1])
            end

            return prim2cons(SVector(H, v, b), equations)
        end

        initial_condition = initial_condition_ec_discontinuous_bottom

        volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
        solver = DGSEM(polydeg = 4,
                       surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
                       volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

        coordinates_min = -1.0
        coordinates_max = 1.0
        mesh = TreeMesh(coordinates_min, coordinates_max,
                        initial_refinement_level = 4,
                        n_cells_max = 10_000)

        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

        tspan = (0.0, 2.0)

        # Get CPU data
        (; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi

        # Get GPU data
        equations_gpu = deepcopy(equations)
        mesh_gpu, solver_gpu, cache_gpu = deepcopy(mesh), deepcopy(solver), deepcopy(cache)
        boundary_conditions_gpu, source_terms_gpu = deepcopy(boundary_conditions),
                                                    deepcopy(source_terms)

        # Set initial time
        t = t_gpu = 0.0

        # Get initial data
        ode = semidiscretize(semi, tspan)
        u_ode = copy(ode.u0)
        du_ode = similar(u_ode)
        u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
        du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

        # Copy data to device
        du_gpu, u_gpu = TrixiCUDA.copy_to_device!(du, u)
        # Reset data on host
        Trixi.reset_du!(du, solver, cache)

        # Test `cuda_volume_integral!`
        TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                        Trixi.have_nonconservative_terms(equations_gpu),
                                        equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                        cache_gpu)
        Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                    equations, solver.volume_integral, solver, cache)
        @test_approx du_gpu ≈ du

        # Test `cuda_prolong2interfaces!`
        TrixiCUDA.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
        Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
        @test_approx cache_gpu.interfaces.u ≈ cache.interfaces.u

        # Test `cuda_interface_flux!`
        TrixiCUDA.cuda_interface_flux!(mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu),
                                       equations_gpu, solver_gpu, cache_gpu)
        Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                                   Trixi.have_nonconservative_terms(equations), equations,
                                   solver.surface_integral, solver, cache)
        @test_approx cache_gpu.elements.surface_flux_values ≈ cache.elements.surface_flux_values

        # Test `cuda_prolong2boundaries!`
        TrixiCUDA.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu, equations_gpu,
                                           cache_gpu)
        Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
        @test_approx cache_gpu.boundaries.u ≈ cache.boundaries.u

        # Test `cuda_boundary_flux!`
        TrixiCUDA.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu,
                                      Trixi.have_nonconservative_terms(equations_gpu),
                                      equations_gpu,
                                      solver_gpu, cache_gpu)
        Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                                  solver.surface_integral, solver)
        @test_approx cache_gpu.elements.surface_flux_values ≈ cache.elements.surface_flux_values

        # Test `cuda_surface_integral!`
        TrixiCUDA.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
        Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral, solver, cache)
        @test_approx du_gpu ≈ du

        # Test `cuda_jacobian!`
        TrixiCUDA.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
        Trixi.apply_jacobian!(du, mesh, equations, solver, cache)
        @test_approx du_gpu ≈ du

        # Test `cuda_sources!`
        TrixiCUDA.cuda_sources!(du_gpu, u_gpu, t_gpu, source_terms_gpu, equations_gpu, cache_gpu)
        Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)
        @test_approx du_gpu ≈ du

        # Copy data back to host
        du_cpu, u_cpu = TrixiCUDA.copy_to_host!(du_gpu, u_gpu)
    end

    @testset "Shallow Water 2D" begin
        equations = ShallowWaterEquations2D(gravity_constant = 9.81)

        initial_condition = initial_condition_weak_blast_wave

        volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
        solver = DGSEM(polydeg = 4,
                       surface_flux = (flux_fjordholm_etal, flux_nonconservative_fjordholm_etal),
                       volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

        coordinates_min = (-1.0, -1.0)
        coordinates_max = (1.0, 1.0)
        mesh = TreeMesh(coordinates_min, coordinates_max,
                        initial_refinement_level = 2,
                        n_cells_max = 10_000)

        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

        tspan = (0.0, 2.0)
        ode = semidiscretize(semi, tspan)

        function initial_condition_ec_discontinuous_bottom(x, t, element_id,
                                                           equations::ShallowWaterEquations2D)
            # Set up polar coordinates
            inicenter = SVector(0.7, 0.7)
            x_norm = x[1] - inicenter[1]
            y_norm = x[2] - inicenter[2]
            r = sqrt(x_norm^2 + y_norm^2)
            phi = atan(y_norm, x_norm)
            sin_phi, cos_phi = sincos(phi)

            # Set the background values
            H = 4.25
            v1 = 0.0
            v2 = 0.0
            b = 0.0

            # Setup the discontinuous water height and velocities
            if element_id == 10
                H = 5.0
                v1 = 0.1882 * cos_phi
                v2 = 0.1882 * sin_phi
            end

            # Setup a discontinuous bottom topography using the element id number
            if element_id == 7
                b = 2.0 + 0.5 * sin(2.0 * pi * x[1]) + 0.5 * cos(2.0 * pi * x[2])
            end

            return prim2cons(SVector(H, v1, v2, b), equations)
        end

        # Point to the data we want to augment
        u = Trixi.wrap_array(ode.u0, semi)
        # Reset the initial condition
        for element in eachelement(semi.solver, semi.cache)
            for j in eachnode(semi.solver), i in eachnode(semi.solver)
                x_node = Trixi.get_node_coords(semi.cache.elements.node_coordinates, equations,
                                               semi.solver, i, j, element)
                u_node = initial_condition_ec_discontinuous_bottom(x_node, first(tspan), element,
                                                                   equations)
                Trixi.set_node_vars!(u, u_node, equations, semi.solver, i, j, element)
            end
        end

        # Get CPU data
        (; mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache) = semi

        # Get GPU data
        equations_gpu = deepcopy(equations)
        mesh_gpu, solver_gpu, cache_gpu = deepcopy(mesh), deepcopy(solver), deepcopy(cache)
        boundary_conditions_gpu, source_terms_gpu = deepcopy(boundary_conditions),
                                                    deepcopy(source_terms)

        # Set initial time
        t = t_gpu = 0.0

        # Get initial data
        # ode = semidiscretize(semi, tspan)
        u_ode = copy(ode.u0)
        du_ode = similar(u_ode)
        # u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)
        du = Trixi.wrap_array(du_ode, mesh, equations, solver, cache)

        # Copy data to device
        du_gpu, u_gpu = TrixiCUDA.copy_to_device!(du, u)
        # Reset data on host
        Trixi.reset_du!(du, solver, cache)

        # Test `cuda_volume_integral!`
        TrixiCUDA.cuda_volume_integral!(du_gpu, u_gpu, mesh_gpu,
                                        Trixi.have_nonconservative_terms(equations_gpu),
                                        equations_gpu, solver_gpu.volume_integral, solver_gpu,
                                        cache_gpu)
        Trixi.calc_volume_integral!(du, u, mesh, Trixi.have_nonconservative_terms(equations),
                                    equations, solver.volume_integral, solver, cache)
        @test_approx du_gpu ≈ du

        # Test `cuda_prolong2interfaces!`
        TrixiCUDA.cuda_prolong2interfaces!(u_gpu, mesh_gpu, equations_gpu, cache_gpu)
        Trixi.prolong2interfaces!(cache, u, mesh, equations, solver.surface_integral, solver)
        @test_approx cache_gpu.interfaces.u ≈ cache.interfaces.u

        # Test `cuda_interface_flux!`
        TrixiCUDA.cuda_interface_flux!(mesh_gpu, Trixi.have_nonconservative_terms(equations_gpu),
                                       equations_gpu, solver_gpu, cache_gpu)
        Trixi.calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                                   Trixi.have_nonconservative_terms(equations), equations,
                                   solver.surface_integral, solver, cache)
        @test_approx cache_gpu.elements.surface_flux_values ≈ cache.elements.surface_flux_values

        # Test `cuda_prolong2boundaries!`
        TrixiCUDA.cuda_prolong2boundaries!(u_gpu, mesh_gpu, boundary_conditions_gpu, equations_gpu,
                                           cache_gpu)
        Trixi.prolong2boundaries!(cache, u, mesh, equations, solver.surface_integral, solver)
        @test_approx cache_gpu.boundaries.u ≈ cache.boundaries.u

        # Test `cuda_boundary_flux!`
        TrixiCUDA.cuda_boundary_flux!(t_gpu, mesh_gpu, boundary_conditions_gpu,
                                      Trixi.have_nonconservative_terms(equations_gpu),
                                      equations_gpu,
                                      solver_gpu, cache_gpu)
        Trixi.calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                                  solver.surface_integral, solver)
        @test_approx cache_gpu.elements.surface_flux_values ≈ cache.elements.surface_flux_values

        # Test `cuda_prolong2mortars!`
        TrixiCUDA.cuda_prolong2mortars!(u_gpu, mesh_gpu, TrixiCUDA.check_cache_mortars(cache_gpu),
                                        solver_gpu, cache_gpu)
        Trixi.prolong2mortars!(cache, u, mesh, equations,
                               solver.mortar, solver.surface_integral, solver)
        @test_approx cache_gpu.mortars.u_upper ≈ cache.mortars.u_upper
        @test_approx cache_gpu.mortars.u_lower ≈ cache.mortars.u_lower

        # Test `cuda_mortar_flux!`
        TrixiCUDA.cuda_mortar_flux!(mesh_gpu, TrixiCUDA.check_cache_mortars(cache_gpu),
                                    Trixi.have_nonconservative_terms(equations_gpu), equations_gpu,
                                    solver_gpu, cache_gpu)
        Trixi.calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                                Trixi.have_nonconservative_terms(equations), equations,
                                solver.mortar, solver.surface_integral, solver, cache)
        @test_approx cache_gpu.elements.surface_flux_values ≈ cache.elements.surface_flux_values

        # Test `cuda_surface_integral!`
        TrixiCUDA.cuda_surface_integral!(du_gpu, mesh_gpu, equations_gpu, solver_gpu, cache_gpu)
        Trixi.calc_surface_integral!(du, u, mesh, equations, solver.surface_integral, solver, cache)
        @test_approx du_gpu ≈ du

        # Test `cuda_jacobian!`
        TrixiCUDA.cuda_jacobian!(du_gpu, mesh_gpu, equations_gpu, cache_gpu)
        Trixi.apply_jacobian!(du, mesh, equations, solver, cache)
        @test_approx du_gpu ≈ du

        # Test `cuda_sources!`
        TrixiCUDA.cuda_sources!(du_gpu, u_gpu, t_gpu, source_terms_gpu, equations_gpu, cache_gpu)
        Trixi.calc_sources!(du, u, t, source_terms, equations, solver, cache)
        @test_approx du_gpu ≈ du

        # Copy data back to host
        du_cpu, u_cpu = TrixiCUDA.copy_to_host!(du_gpu, u_gpu)
    end
end

end # module
