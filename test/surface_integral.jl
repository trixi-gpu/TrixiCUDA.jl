module SurfaceIntegralTests

using CUDA, Test, Trixi, TrixiCUDA

# Accumulate each face independently, without the kernels' flattened node indexing.
function reference_surface_integral!(du, surface_flux_values, inverse_weights, ndims_)
    for index in CartesianIndices(du)
        variable = index[1]
        nodes = ntuple(axis -> index[axis + 1], ndims_)
        element = index[ndims_ + 2]
        for axis in 1:ndims_
            transverse = Tuple(nodes[other] for other in 1:ndims_ if other != axis)
            if nodes[axis] == 1
                du[index] -= surface_flux_values[variable, transverse..., 2 * axis - 1, element] *
                             first(inverse_weights)
            end
            if nodes[axis] == size(du, axis + 1)
                du[index] += surface_flux_values[variable, transverse..., 2 * axis, element] *
                             last(inverse_weights)
            end
        end
    end
    return du
end

@testset "Surface integrals use inverse quadrature weights" begin
    equation_types = (CompressibleEulerEquations1D, CompressibleEulerEquations2D,
                      CompressibleEulerEquations3D)
    for ndims_ in 1:3, RealT in (Float32, Float64), polydeg in (1, 3, 4)
        @testset "$ndims_ dimensions, $RealT, degree $polydeg" begin
            basis = LobattoLegendreBasisGPU(RealT, polydeg)
            basis_cpu = LobattoLegendreBasis(RealT, polydeg)
            @test !hasproperty(basis, :boundary_interpolation)
            @test Array(basis.inverse_weights) == basis_cpu.inverse_weights
            solver = DGSEMGPU(basis)
            equations = equation_types[ndims_](RealT(1.4))
            mesh = TreeMesh(ntuple(_ -> -one(RealT), ndims_), ntuple(_ -> one(RealT), ndims_),
                            initial_refinement_level = 1,
                            periodicity = true,
                            RealT = RealT)

            nnodes_ = polydeg + 1
            nelements_ = 2^ndims_
            shape = (Trixi.nvariables(equations), ntuple(_ -> nnodes_, ndims_)..., nelements_)
            face_shape = (shape[1], ntuple(_ -> nnodes_, ndims_ - 1)..., 2 * ndims_, nelements_)
            initial_du = reshape(sin.(RealT.(1:prod(shape))), shape)
            surface_flux_values = reshape(cos.(RealT.(1:prod(face_shape))), face_shape)
            cache = (; elements = (; surface_flux_values = CuArray(surface_flux_values)))

            for asymmetric_weights in (false, true)
                # Distinct endpoints also catch accidentally reading an interior weight.
                inverse_weights = collect(basis_cpu.inverse_weights)
                if asymmetric_weights
                    inverse_weights[1] = RealT(7)
                    inverse_weights[end] = RealT(11)
                end
                copyto!(basis.inverse_weights, inverse_weights)
                expected = reference_surface_integral!(copy(initial_du), surface_flux_values,
                                                       inverse_weights, ndims_)
                du = CuArray(initial_du)
                TrixiCUDA.cuda_surface_integral!(du, mesh, equations, solver, cache)
                @test isapprox(Array(du), expected; rtol = 20 * eps(RealT), atol = 20 * eps(RealT))
            end
        end
    end
end

end # module
