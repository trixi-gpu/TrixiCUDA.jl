# Compare initialization and the complete DG RHS without running output callbacks.
function test_shallowwater_rhs(semi;
                              boundary_conditions = boundary_condition_periodic,
                              initialize! = nothing)
    solver = semi.solver
    solver_gpu = DGSEMGPU(polydeg = polydeg(solver),
                          surface_flux = solver.surface_integral.surface_flux,
                          volume_integral = solver.volume_integral)
    semi_gpu = SemidiscretizationHyperbolicGPU(semi.mesh, semi.equations,
                                               semi.initial_condition, solver_gpu;
                                               boundary_conditions = boundary_conditions,
                                               source_terms = semi.source_terms)
    tspan = (0.0, 0.1)
    ode = semidiscretize(semi, tspan)
    ode_gpu = semidiscretizeGPU(semi_gpu, tspan)
    u = Trixi.wrap_array(ode.u0, semi)
    u_host = Array(ode_gpu.u0)
    @test all(isfinite, u)
    @test all(isfinite, u_host)
    @test u_host ≈ u

    if initialize! !== nothing
        initialize!(u, semi)
        initialize!(u_host, semi)
        # Updating the host snapshot alone does not update the GPU solution.
        copyto!(ode_gpu.u0, u_host)
        @test Array(ode_gpu.u0) ≈ u
    end

    du = similar(ode.u0)
    du_gpu = similar(ode_gpu.u0)
    # A nonzero time also exercises the time-dependent manufactured source terms.
    for t in (0.0, 0.1)
        ode.f(du, ode.u0, ode.p, t)
        ode_gpu.f(du_gpu, ode_gpu.u0, ode_gpu.p, t)
        du_host = Array(du_gpu)
        du_reference = Trixi.wrap_array(du, semi)
        @test all(isfinite, du_reference)
        @test all(isfinite, du_host)
        @test isapprox(du_host, du_reference; rtol = 1.0e-10, atol = 1.0e-11)
        # Bottom topography is a stationary component of the shallow-water state.
        @test all(iszero, selectdim(du_host, 1, size(du_host, 1)))
    end
    return nothing
end
