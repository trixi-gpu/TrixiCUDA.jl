module TestAPI

using Test
using CUDA
using Trixi, TrixiCUDA

# Requiring a matching time type also checks conversion before GPU initialization.
struct TypedInitialCondition{T} end
(::TypedInitialCondition{T})(x, t::T, equations) where {T} = SVector(sinpi(x[1]) + t)

@testset "Upstream API consistency" begin
    @test_throws ArgumentError DGSEMGPU(polydeg = 2, basis_type = LobattoLegendreBasis)

    for RealT in (Float32, Float64)
        @testset "$RealT" begin
            equations = LinearScalarAdvectionEquation1D(one(RealT))
            basis_factory = (T, degree) -> LobattoLegendreBasisGPU(T, degree)
            solver = DGSEMGPU(RealT = RealT, polydeg = 2, basis_type = basis_factory,
                              surface_flux = flux_lax_friedrichs)
            @test solver.basis isa LobattoLegendreBasisGPU
            @test real(solver) === RealT

            mesh = TreeMesh(-one(RealT), one(RealT), initial_refinement_level = 2,
                            periodicity = true, RealT = RealT)
            initial_condition = TypedInitialCondition{RealT}()
            @test_throws UndefKeywordError SemidiscretizationHyperbolicGPU(mesh, equations,
                                                                           initial_condition, solver)
            semi = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver;
                                                   boundary_conditions = boundary_condition_periodic)
            @test real(semi) === RealT
            @test semi.cache_gpu.elements isa TrixiCUDA.TreeElementContainerGPU1D
            @test semi.cache_gpu.interfaces isa TrixiCUDA.TreeInterfaceContainerGPU1D
            @test semi.cache_gpu.boundaries isa TrixiCUDA.TreeBoundaryContainerGPU1D

            ode = semidiscretizeGPU(semi, (0.25, 1))
            @test ode.tspan === (RealT(0.25), one(RealT))
            @test eltype(ode.u0) === RealT
            @test ode.f.f === TrixiCUDA.rhs_hyperbolic!

            # Exercise the ODE-facing RHS, including backend selection and array wrapping.
            du = similar(ode.u0)
            ode.f(du, ode.u0, semi, first(ode.tspan))
            solver_cpu = DGSEM(RealT = RealT, polydeg = 2, surface_flux = flux_lax_friedrichs)
            semi_cpu = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver_cpu;
                                                    boundary_conditions = boundary_condition_periodic)
            ode_cpu = semidiscretize(semi_cpu, (0.25, 1))
            du_cpu = similar(ode_cpu.u0)
            ode_cpu.f(du_cpu, ode_cpu.u0, semi_cpu, first(ode_cpu.tspan))
            @test vec(Array(du))≈du_cpu rtol=100*eps(RealT) atol=100*eps(RealT)

            # Check ODE metadata without assuming GPU support for an implicit solver.
            n = length(ode.u0)
            jac_prototype = ones(Float64, n, n)
            colorvec = collect(1:n)
            ode_jac = semidiscretizeGPU(semi, (0, 1); jac_prototype, colorvec)
            @test eltype(ode_jac.f.jac_prototype) === RealT
            @test ode_jac.f.jac_prototype isa Matrix
            @test ode_jac.f.jac_prototype == RealT.(jac_prototype)
            @test ode_jac.f.colorvec == colorvec
            @test ode_jac.tspan === (zero(RealT), one(RealT))

            indicator = IndicatorHennemannGassner(equations, solver.basis; variable = first)
            hg = VolumeIntegralShockCapturingHG(indicator)
            @test DGSEMGPU(solver.basis, solver.surface_integral, hg).volume_integral === hg
            generalized_hg = Trixi.VolumeIntegralShockCapturingHGType(indicator;
                                                                      volume_integral_default = VolumeIntegralWeakForm(),
                                                                      volume_integral_blend_high_order = hg.volume_integral_blend_high_order,
                                                                      volume_integral_blend_low_order = hg.volume_integral_blend_low_order)
            @test_throws ArgumentError DGSEMGPU(RealT = RealT, polydeg = 2,
                                                volume_integral = generalized_hg)
        end
    end
end

end # module
