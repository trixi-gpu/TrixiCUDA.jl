# Run by the existing Buildkite Julia test matrix via test/runtests.jl.
# Reports are saved under benchmark/results/coefficient_inlining/.
# Timings include kernel launch and CUDA synchronization, after warmup.
module BenchmarkCoefficientInlining

using BenchmarkTools, CUDA, Trixi, TrixiCUDA
using Printf, Statistics

CUDA.functional() || error("A functional CUDA device is required")
CUDA.allowscalar(false)

launch_kernel!(compiled, arguments, configuration) = compiled(arguments...; configuration...)

function benchmark_case(::Type{RealT}, degree, family, raw) where {RealT}
    equations = family == "advection" ?
                LinearScalarAdvectionEquation3D((RealT(0.2), RealT(0.4), RealT(0.6))) :
                CompressibleEulerEquations3D(RealT(1.4))
    initial_condition = family == "advection" ? initial_condition_convergence_test :
                        initial_condition_weak_blast_wave
    solver = DGSEM(polydeg = degree, RealT = RealT)
    # 4096 elements, with the same input for both compiler settings.
    mesh = TreeMesh(ntuple(_ -> -RealT(2), 3), ntuple(_ -> RealT(2), 3),
                    initial_refinement_level = 4, periodicity = true, RealT = RealT)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                        boundary_conditions = boundary_condition_periodic)
    t = zero(RealT)
    ode = semidiscretize(semi, (t, RealT(0.1)))
    reference = Array(Trixi.wrap_array(ode.u0, mesh, equations, solver, semi.cache))
    coordinates = CuArray(semi.cache.elements.node_coordinates)
    state = CuArray{RealT}(undef, size(reference)...)
    arguments = (state, coordinates, initial_condition, t, equations)
    kernels, configurations = Dict(), Dict()
    measurements = Dict(false => Float64[], true => Float64[])

    # Compile both variants and validate them before any timing.
    for force_inline in (false, true)
        compiled = CUDA.@cuda launch=false always_inline=force_inline TrixiCUDA.compute_coefficients_kernel!(state, coordinates, initial_condition, t,
                                                                                                             equations)
        configuration = TrixiCUDA.kernel_configurator_2d(compiled, size(state, 2)^3,
                                                         size(state, 5))
        kernels[force_inline] = compiled
        configurations[force_inline] = configuration
        fill!(state, RealT(NaN))
        for _ in 1:5
            launch_kernel!(compiled, arguments, configuration)
        end
        CUDA.synchronize()
        result = Array(state)
        @assert all(isfinite, result)
        @assert isapprox(result, reference; rtol = 32 * eps(RealT), atol = 32 * eps(RealT))
    end

    # Alternate the order to reduce bias from warmup or changing GPU clocks.
    for round in 1:3
        for force_inline in (isodd(round) ? (false, true) : (true, false))
            compiled = kernels[force_inline]
            configuration = configurations[force_inline]
            trial = @benchmark (CUDA.@sync launch_kernel!($compiled, $arguments, $configuration)) evals=1 samples=2000 seconds=1.0
            time_ns = median(trial).time
            push!(measurements[force_inline], time_ns)
            println(raw, join((3, family, RealT, degree, size(state, 5), force_inline, round,
                               time_ns, minimum(trial).time, length(trial)), ','))
            flush(raw)
        end
    end
    spread(values) = 100 * (maximum(values) - minimum(values)) / median(values)
    inlined = median(measurements[true])
    baseline = median(measurements[false])
    row = @sprintf("| %dD | %s | %s | %d | %.3f | %.3f | %.3fx | %+.2f%% | %.1f / %.1f |",
                   3, family, RealT, degree, baseline / 1e3, inlined / 1e3,
                   baseline / inlined, 100 * (inlined / baseline - 1),
                   spread(measurements[false]), spread(measurements[true]))
    println(row)
    config_row = "3D $family $RealT degree=$degree: false=$(get(configurations, false, nothing)), true=$(configurations[true])"
    return row, config_row
end

function main()
    output = joinpath(@__DIR__, "results", "coefficient_inlining")
    mkpath(output)
    open(joinpath(output, "environment.txt"), "w") do io
        println(io, "Commit: ", get(ENV, "BUILDKITE_COMMIT", "local"))
        println(io, "Build: ", get(ENV, "BUILDKITE_BUILD_URL", "local"))
        println(io, "Julia: ", VERSION)
        CUDA.versioninfo(io)
        for package in (CUDA, Trixi, TrixiCUDA, BenchmarkTools, Statistics)
            println(io, nameof(package), ": ", pkgversion(package))
        end
    end
    println("Julia: ", VERSION, " | GPU: ", CUDA.name(CUDA.device()))
    rows, configurations, failures = String[], String[], String[]
    open(joinpath(output, "trials.csv"), "w") do raw
        println(raw, "dimension,family,precision,degree,elements,always_inline,round,median_ns,minimum_ns,samples")
        for RealT in (Float32, Float64)
            for family in ("advection", "Euler"), degree in (3, 7)
                name = "3D $family $RealT degree=$degree"
                println("Benchmarking ", name)
                flush(stdout)
                try
                    row, configuration = benchmark_case(RealT, degree, family, raw)
                    push!(rows, row)
                    push!(configurations, configuration)
                catch err
                    err isa InterruptException && rethrow()
                    push!(failures, name)
                    open(joinpath(output, "errors.txt"), "a") do io
                        println(io, name)
                        showerror(io, err, catch_backtrace())
                        println(io)
                    end
                    @error "Benchmark case failed" name exception=(err, catch_backtrace())
                end
                GC.gc()
                CUDA.reclaim()
            end
        end
    end
    report = joinpath(output, "report.md")
    open(report, "w") do io
        println(io, "# 3D coefficient initialization: forced inlining")
        println(io, "\nJulia $VERSION; commit `", get(ENV, "BUILDKITE_COMMIT", "local"), "`.")
        println(io,
                "\nPrecompiled kernel launch + CUDA synchronization only (BenchmarkTools + CUDA.@sync); no compilation, array allocation, mesh construction, or CPU/GPU copies inside timing. Each variant uses the production launch configurator. Five warmup launches; three alternating rounds, evals=1, up to 2000 samples/1 second per round. Reported times are medians of round medians. Each compiled variant must match CPU initial coefficients before timing. Only the real 3D compute_coefficients_kernel! is measured; both compiler settings must compile and pass validation.")
        println(io,
                "\n| Dimension | Initial condition | Precision | Degree | inline=false (us) | inline=true (us) | Speedup | Runtime change | Round spread %, false / true |")
        println(io, "|---|---|---|---:|---:|---:|---:|---:|---:|")
        foreach(row -> println(io, row), rows)
        println(io,
                "\nSpeedup = false/true; negative runtime change means faster. Round spread = (max-min)/median. Small differences comparable to the spread are inconclusive. This measures initialization only, not RHS or whole-simulation speedup.")
        println(io, "\n## Launch configurations\n")
        foreach(row -> println(io, "- ", row), configurations)
        println(io, "\nFailed cases: ", isempty(failures) ? "none" : join(failures, "; "))
    end
    println(read(report, String))
    isempty(failures) || error("Some cases failed; see benchmark artifacts")
end

main()

end # module BenchmarkCoefficientInlining
