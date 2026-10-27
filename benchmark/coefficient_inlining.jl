# Run by the existing Buildkite Julia test matrix via test/runtests.jl.
# Reports are saved under benchmark/results/coefficient_inlining/.
# Timings include kernel launch and CUDA synchronization, after warmup.
module BenchmarkCoefficientInlining

using BenchmarkTools, CUDA, Trixi, TrixiCUDA
using Pkg, Printf, Statistics

CUDA.functional() || error("A functional CUDA device is required")
CUDA.allowscalar(false)

launch_kernel!(compiled, arguments, configuration) = compiled(arguments...; configuration...)

function make_equations(nd, RealT, family)
    if family == "advection"
        types = (LinearScalarAdvectionEquation1D, LinearScalarAdvectionEquation2D,
                 LinearScalarAdvectionEquation3D)
        velocity = ntuple(i -> RealT(0.2 * i), nd)
        return types[nd](nd == 1 ? only(velocity) : velocity)
    elseif family == "Euler"
        types = (CompressibleEulerEquations1D, CompressibleEulerEquations2D,
                 CompressibleEulerEquations3D)
        return types[nd](RealT(1.4))
    else
        components = family == "multicomponent3" ? 3 : 4
        types = (CompressibleEulerMulticomponentEquations1D,
                 CompressibleEulerMulticomponentEquations2D)
        return types[nd](gammas = ntuple(i -> RealT(1.4 + 0.01 * i), components),
                         gas_constants = ntuple(i -> RealT(0.4 + 0.01 * i), components))
    end
end

function benchmark_case(nd, ::Type{RealT}, degree, family, raw, output) where {RealT}
    equations = make_equations(nd, RealT, family)
    initial_condition = family == "advection" ? initial_condition_convergence_test :
                        initial_condition_weak_blast_wave
    solver = DGSEM(polydeg = degree, RealT = RealT)
    # 4096 elements in every dimension; avoid comparing different problem sizes.
    level = (12, 6, 4)[nd]
    mesh = TreeMesh(ntuple(_ -> -RealT(2), nd), ntuple(_ -> RealT(2), nd),
                    initial_refinement_level = level, periodicity = true, RealT = RealT)
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
        compiled = try
            CUDA.@cuda launch=false always_inline=force_inline TrixiCUDA.compute_coefficients_kernel!(
                state, coordinates, initial_condition, t, equations)
        catch err
            err isa InterruptException && rethrow()
            (force_inline || nd == 3) && rethrow()
            # Skip 1D/2D comparisons without a compilable baseline; 3D remains
            # covered independently, including on older Julia versions.
            open(joinpath(output, "baseline_compile_errors.txt"), "a") do io
                println(io, "$(nd)D $family $RealT degree=$degree: inline=false COMPILE_FAILED")
                showerror(io, err, catch_backtrace())
                println(io)
            end
            println("Skipping this 1D/2D comparison: inline=false cannot compile")
            row = "| $(nd)D | $family | $RealT | $degree | COMPILE_FAILED | SKIPPED | N/A | N/A | N/A |"
            return row, "$(nd)D $family $RealT degree=$degree: skipped (baseline compilation failed)"
        end
        configuration = TrixiCUDA.kernel_configurator_2d(compiled, size(state, 2)^nd,
                                                         size(state, nd + 2))
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
            haskey(kernels, force_inline) || continue
            compiled = kernels[force_inline]
            configuration = configurations[force_inline]
            trial = @benchmark (CUDA.@sync launch_kernel!($compiled, $arguments, $configuration)) evals=1 samples=2000 seconds=1.0
            time_ns = median(trial).time
            push!(measurements[force_inline], time_ns)
            println(raw, join((nd, family, RealT, degree, 4096, force_inline, round,
                               time_ns, minimum(trial).time, length(trial)), ','))
            flush(raw)
        end
    end
    spread(values) = 100 * (maximum(values) - minimum(values)) / median(values)
    inlined = median(measurements[true])
    baseline = median(measurements[false])
    row = @sprintf("| %dD | %s | %s | %d | %.3f | %.3f | %.3fx | %+.2f%% | %.1f / %.1f |",
                   nd, family, RealT, degree, baseline / 1e3, inlined / 1e3,
                   baseline / inlined, 100 * (inlined / baseline - 1),
                   spread(measurements[false]), spread(measurements[true]))
    println(row)
    config_row = "$(nd)D $family $RealT degree=$degree: false=$(get(configurations, false, nothing)), true=$(configurations[true])"
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
        Pkg.status(; io = io, mode = Pkg.PKGMODE_MANIFEST)
    end
    println("Julia: ", VERSION, " | GPU: ", CUDA.name(CUDA.device()))
    rows, configurations, failures = String[], String[], String[]
    open(joinpath(output, "trials.csv"), "w") do raw
        println(raw, "dimension,family,precision,degree,elements,always_inline,round,median_ns,minimum_ns,samples")
        for nd in 1:3, RealT in (Float32, Float64)
            cases = [(family, degree) for family in ("advection", "Euler") for degree in (3, 7)]
            if nd <= 2
                append!(cases, [("multicomponent3", 3), ("multicomponent4", 3)])
            end
            for (family, degree) in cases
                name = "$(nd)D $family $RealT degree=$degree"
                println("Benchmarking ", name)
                flush(stdout)
                try
                    row, configuration = benchmark_case(nd, RealT, degree, family, raw, output)
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
        println(io, "# Coefficient initialization: forced inlining")
        println(io, "\nJulia $VERSION; commit `", get(ENV, "BUILDKITE_COMMIT", "local"), "`.")
        println(io, "\nPrecompiled kernel launch + CUDA synchronization only (BenchmarkTools + CUDA.@sync); no compilation, array allocation, mesh construction, or CPU/GPU copies inside timing. Each variant uses the production launch configurator. Five warmup launches; three alternating rounds, evals=1, up to 2000 samples/1 second per round. Reported times are medians of round medians. Each compiled variant must match CPU initial coefficients before timing. If a 1D/2D inline=false baseline cannot compile, skip that comparison (COMPILE_FAILED/SKIPPED, no inferred speedup); 3D comparisons run independently.")
        println(io, "\n| Dimension | Initial condition | Precision | Degree | inline=false (us) | inline=true (us) | Speedup | Runtime change | Round spread %, false / true |")
        println(io, "|---|---|---|---:|---:|---:|---:|---:|---:|")
        foreach(row -> println(io, row), rows)
        println(io, "\nSpeedup = false/true; negative runtime change means faster. Round spread = (max-min)/median. Small differences comparable to the spread are inconclusive. This measures initialization only, not RHS or whole-simulation speedup.")
        println(io, "\n## Launch configurations\n")
        foreach(row -> println(io, "- ", row), configurations)
        println(io, "\nFailed cases: ", isempty(failures) ? "none" : join(failures, "; "))
    end
    println(read(report, String))
    isempty(failures) || error("Some cases failed; see benchmark artifacts")
end

main()

end # module BenchmarkCoefficientInlining
