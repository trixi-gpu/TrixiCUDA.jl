module TestMulticomponentSaveSolution

using CUDA, HDF5, Trixi, TrixiCUDA
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using SciMLBase: init, step!
using Test

function check_saved_solution(filename, u, equations, time)
    states = reshape(u, nvariables(equations), :)
    expected = hcat((cons2prim(view(states, :, i), equations) for i in axes(states, 2))...)
    @test isfile(filename)
    h5open(filename, "r") do file
        @test read(attributes(file)["time"]) == time
        @test read(attributes(file)["n_vars"]) == size(expected, 1)
        for v in axes(expected, 1)
            @test read(file["variables_$v"]) ≈ expected[v, :]
        end
    end
end

@testset "Multicomponent solution saving" begin
    CUDA.allowscalar(false)
    for nd in (1, 2), ncomponents in (3, 4)
        @testset "$(nd)D, $ncomponents components" begin
            equations_type = nd == 1 ? CompressibleEulerMulticomponentEquations1D :
                             CompressibleEulerMulticomponentEquations2D
            equations = equations_type(gammas = ntuple(_ -> 1.4, ncomponents),
                                       gas_constants = ntuple(_ -> 0.4, ncomponents))
            solver = DGSEMGPU(polydeg = 3, surface_flux = flux_ranocha,
                              volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))
            mesh = TreeMesh(ntuple(_ -> -2.0, nd), ntuple(_ -> 2.0, nd),
                            initial_refinement_level = 1, periodicity = true)
            semi = SemidiscretizationHyperbolicGPU(mesh, equations,
                                                  initial_condition_weak_blast_wave, solver)
            ode = semidiscretizeGPU(semi, (0.0, 2.0e-4))
            initial = Array(ode.u0)

            mktempdir() do output_directory
                callback = SaveSolutionCallback(interval = 1,
                                                save_initial_solution = true,
                                                save_final_solution = true,
                                                output_directory = output_directory,
                                                solution_variables = cons2prim)
                integrator = init(ode, CarpenterKennedy2N54(williamson_condition = false);
                                  dt = 1.0e-4, adaptive = false, callback = callback,
                                  save_everystep = false)
                @test Array(integrator.u) == initial
                check_saved_solution(joinpath(output_directory, "solution_000000000.h5"),
                                     initial, equations, 0.0)

                step!(integrator)
                @test integrator.t > 0
                @test all(isfinite, Array(integrator.u))
                check_saved_solution(joinpath(output_directory, "solution_000000001.h5"),
                                     Array(integrator.u), equations, integrator.t)
            end
        end
    end
end

end # module
