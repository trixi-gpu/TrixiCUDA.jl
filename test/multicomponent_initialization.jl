module TestMulticomponentInitialization

using CUDA, Trixi, TrixiCUDA
using Test

@testset "Multicomponent Euler initialization" begin
    for ndims in (1, 2), RealT in (Float32, Float64), ncomponents in (3, 4)
        @testset "$ndims dimensions, $RealT, $ncomponents components" begin
            equations_type = ndims == 1 ? CompressibleEulerMulticomponentEquations1D :
                             CompressibleEulerMulticomponentEquations2D
            equations = equations_type(gammas = ntuple(i -> RealT(1.4 + 0.01 * i), ncomponents),
                                       gas_constants = ntuple(i -> RealT(0.4 + 0.01 * i), ncomponents))
            initial_condition = initial_condition_weak_blast_wave
            solver = DGSEM(polydeg = 3, RealT = RealT)
            solver_gpu = DGSEMGPU(polydeg = 3, RealT = RealT)
            mesh = TreeMesh(ntuple(_ -> -RealT(2), ndims), ntuple(_ -> RealT(2), ndims),
                            initial_refinement_level = 2,
                            periodicity = true,
                            RealT = RealT)
            semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                                boundary_conditions = boundary_condition_periodic)
            semi_gpu = SemidiscretizationHyperbolicGPU(mesh, equations, initial_condition, solver_gpu)
            tspan = (zero(RealT), RealT(0.1))
            ode = semidiscretize(semi, tspan)
            ode_gpu = semidiscretizeGPU(semi_gpu, tspan)
            reference = Trixi.wrap_array(ode.u0, mesh, equations, solver, semi.cache)
            result = Array(ode_gpu.u0)

            @test eltype(result) == RealT
            @test all(isfinite, result)
            @test isapprox(result, reference; rtol = 32 * eps(RealT), atol = 32 * eps(RealT))
        end
    end
end

end # module
