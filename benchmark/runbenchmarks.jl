using TrixiCUDA, CUDA, BenchmarkTools
using Dates, JSON, Pkg, Printf, SHA, TOML
include("cases.jl")

const RESULT_SCHEMA = 3
const TRIXICUDA_UUID = "5056661c-942a-472c-b3bc-01507f717a60"

function measurement_context(result)
    env = result["environment"]
    fields = ("julia", "gpu", "gpu_uuid", "host", "runtime", "driver", "threads", "harness")
    context = Dict{String, Any}(field => env[field] for field in fields)
    context["packages"] = filter(p -> first(p) != TRIXICUDA_UUID, env["packages"])
    context["parameters"] = result["parameters"]
    return context
end

function validate_result(result)
    result["schema"] == RESULT_SCHEMA || error("Unknown result schema")
    isempty(result["benchmarks"]) && error("No benchmark results")
    for (name, case) in result["benchmarks"]
        time = case["minimum_ns"]
        isfinite(time) && time > 0 || error("Invalid timing: $name")
        times = case["samples_ns"]
        length(times) == case["samples"] > 0 || error("Invalid sample count: $name")
        all(t -> isfinite(t) && t > 0, times) || error("Invalid samples: $name")
        minimum(times) == time || error("Minimum does not match samples: $name")
    end
    return nothing
end

function validate_comparison(baseline, target)
    validate_result(baseline)
    validate_result(target)
    measurement_context(baseline) == measurement_context(target) ||
        error("Benchmark environments, dependencies, harnesses, or parameters differ")
    b, t = baseline["benchmarks"], target["benchmarks"]
    keys(b) == keys(t) || error("Benchmark case sets differ")
    for name in keys(t), field in ("shape", "seed", "dofs_per_field")
        b[name][field] == t[name][field] || error("Input mismatch in $name: $field")
    end
    return nothing
end

function write_result(path, result)
    mkpath(dirname(path))
    # An interrupted job must never leave a partially written cache entry.
    temp, io = mktemp(dirname(path))
    try
        JSON.json(io, result)
        close(io)
        mv(temp, path; force = true)
    finally
        close(io)
        rm(temp; force = true)
    end
end

read_result(path) = JSON.parsefile(path; dicttype = Dict{String, Any})

function cache_path(directory, result)
    identity = measurement_context(result)
    identity["schema"] = RESULT_SCHEMA
    identity["commit"] = result["environment"]["commit"]
    key = bytes2hex(sha256(sprint(io -> TOML.print(io, identity; sorted = true))))
    return joinpath(directory, key * ".json")
end

function restore_baseline(path, expected, case_names)
    isfile(path) || return nothing
    try
        result = read_result(path)
        validate_result(result)
        result["environment"]["commit"] == expected["environment"]["commit"] || return nothing
        measurement_context(result) == measurement_context(expected) || return nothing
        Set(keys(result["benchmarks"])) == Set(case_names) || return nothing
        result["reused"] = true
        return result
    catch exception
        @warn "Ignoring invalid baseline cache" path exception
        return nothing
    end
end

function benchmark_rhs(case; seconds, samples, seed)
    semi = make_problem(case)
    (; mesh, equations, boundary_conditions, source_terms, solver, cache_gpu, cache_cpu) = semi
    u_host = random_input(case, equations, solver, cache_gpu, seed)
    u = CuArray(u_host)
    du = CUDA.zeros(Float32, size(u))
    t = 0.0f0

    # Compile and populate caches before measuring. Fail on invalid random states
    # instead of publishing plausible timings for NaN-producing computations.
    for _ in 1:2
        fill!(du, 0.0f0)
        CUDA.@sync TrixiCUDA.rhs_gpu!(du, u, t, mesh, equations, boundary_conditions,
                                      source_terms, solver, cache_gpu, cache_cpu)
    end
    all(isfinite, Array(du)) || error("Non-finite RHS in $(case.name)")

    # Synchronize the reset outside the timed expression. evals=1 guarantees each
    # measurement starts with du=0, including RHS implementations that accumulate.
    # The block keeps BenchmarkTools options outside CUDA.@sync.
    trial = @benchmark begin
        CUDA.@sync TrixiCUDA.rhs_gpu!($du, $u, $t, $mesh, $equations,
                                      $boundary_conditions, $source_terms,
                                      $solver, $cache_gpu, $cache_cpu)
    end setup=(CUDA.@sync fill!($du, 0.0f0)) evals=1 seconds=seconds samples=samples
    return Dict("minimum_ns" => minimum(trial).time,
                "samples_ns" => trial.times,
                "samples" => length(trial),
                "dofs_per_field" => length(u) ÷ Trixi.nvariables(equations),
                "shape" => collect(size(u)), "seed" => seed)
end

function measure()
    (length(ARGS) == 2 || (length(ARGS) == 3 && ARGS[3] == "--reuse")) ||
        error("Usage: runbenchmarks.jl OUTPUT.json COMMIT_SHA [--reuse]")
    CUDA.functional() || error("A functional CUDA GPU is required for RHS benchmarks")
    seconds = parse(Float64, get(ENV, "TRIXICUDA_BENCHMARK_SECONDS", "2"))
    samples = parse(Int, get(ENV, "TRIXICUDA_BENCHMARK_SAMPLES", "100"))
    seconds > 0 && samples > 0 || error("Benchmark duration and samples must be positive")
    filter = get(ENV, "TRIXICUDA_BENCHMARK_FILTER", "")
    cases = [c for c in rhs_cases() if occursin(filter, c.name)]
    isempty(cases) && error("No benchmark cases match '$filter'")
    # This setting is also used by the shock-capturing path, which copies to CPU.
    Threads.nthreads() == 1 || error("Run with --threads=1 for reproducible comparisons")
    packages = Dict(string(id) => Dict("name" => info.name, "version" => string(info.version),
                                       "tree" => string(info.tree_hash))
                    for (id, info) in Pkg.dependencies() if info.version !== nothing)
    # Include all code and project settings that can change the measured workload.
    files = Dict(file => read(joinpath(@__DIR__, file), String)
                 for file in ("cases.jl", "runbenchmarks.jl", "Project.toml"))
    harness = bytes2hex(sha256(sprint(io -> TOML.print(io, files; sorted = true))))
    environment = Dict("commit" => ARGS[2], "julia" => string(VERSION),
                       "gpu" => CUDA.name(CUDA.device()), "gpu_uuid" => string(CUDA.uuid(CUDA.device())),
                       "host" => gethostname(), "harness" => harness,
                       "runtime" => string(CUDA.runtime_version()),
                       "driver" => CUDA.has_nvml() ? string(CUDA.NVML.driver_version()) : string(CUDA.driver_version()),
                       "threads" => Threads.nthreads(), "packages" => packages,
                       "cuda_versioninfo" => sprint(CUDA.versioninfo),
                       "agent" => get(ENV, "BUILDKITE_AGENT_NAME", "local"),
                       "measured_at" => string(now(UTC)),
                       "build_url" => get(ENV, "BUILDKITE_BUILD_URL", "local"))
    output = Dict("schema" => RESULT_SCHEMA, "environment" => environment,
                  "parameters" => Dict("seconds" => seconds, "samples" => samples,
                                       "precision" => "Float32", "evals" => 1,
                                       "filter" => filter), "reused" => false)
    # JuliaCI/julia preserves scratchspaces between jobs on this agent. Cache
    # entries include the device UUID, host, exact source SHA and full context.
    directory = get(ENV, "TRIXICUDA_BENCHMARK_CACHE",
                    joinpath(first(DEPOT_PATH), "scratchspaces", TRIXICUDA_UUID, "rhs-benchmarks"))
    cached_file = cache_path(directory, output)
    if length(ARGS) == 3
        cached = restore_baseline(cached_file, output, [case.name for case in cases])
        if cached !== nothing
            @info "Reusing baseline benchmark" commit=ARGS[2] measured_at=cached["environment"]["measured_at"]
            write_result(ARGS[1], cached)
            return nothing
        end
        @info "No matching baseline measurement; benchmarking" commit = ARGS[2]
    end
    results = Dict{String, Any}()
    for case in cases
        @info "Benchmarking $(case.name)"
        results[case.name] = benchmark_rhs(case; seconds, samples, seed = 123)
        # Avoid retaining all case allocations on small benchmark GPUs.
        GC.gc(true)
        CUDA.reclaim()
    end
    output["benchmarks"] = results
    validate_result(output)
    write_result(ARGS[1], output)
    write_result(cached_file, output)
    return nothing
end

function write_report(io, baseline, target)
    validate_comparison(baseline, target)
    b_env, t_env = baseline["environment"], target["environment"]
    b, t = baseline["benchmarks"], target["benchmarks"]
    speedup = exp(sum(log(b[name]["minimum_ns"] / t[name]["minimum_ns"]) for name in keys(t)) / length(t))
    println(io, "## GPU RHS benchmark comparison\n")
    println(io, "Baseline: `", b_env["commit"], "`  \nTarget: `", t_env["commit"], "`\n")
    println(io, "GPU: ", t_env["gpu"], "; Julia ", t_env["julia"],
            "; CUDA runtime ", t_env["runtime"], "; driver ", t_env["driver"], ".\n")
    println(io, "Minimum wall time with GPU synchronization; Float32, one Julia thread. ",
            "Compilation, input generation, initial input transfers, and resetting du are excluded. ",
            "CPU work and transfers inside the RHS are included.\n")
    @printf(io, "Cases: %d; geometric mean speedup: **%.3fx**.\n\n", length(t), speedup)
    println(io, "| RHS case | DOFs/field | Baseline (ms) | Target (ms) | Time change | Speedup |")
    println(io, "|:--|--:|--:|--:|--:|--:|")
    for name in sort!(collect(keys(t)))
        base, head = b[name], t[name]
        bt, tt = base["minimum_ns"], head["minimum_ns"]
        @printf(io, "| %s | %d | %.4f | %.4f | %+.2f%% | %.3fx |\n",
                name, head["dofs_per_field"], bt / 1e6, tt / 1e6,
                100 * (tt / bt - 1), bt / tt)
    end
    println(io, "\nNegative time change and speedup > 1 indicate improvement. ",
            "Informational only: no regression threshold is applied. ",
            "Small differences may be measurement noise.\n")
    for (label, result) in (("Baseline", baseline), ("Target", target))
        env = result["environment"]
        println(io, label, " measurement: ", result["reused"] ? "reused" : "fresh",
                "; ", env["measured_at"], " UTC; ", env["build_url"], ".\n")
    end
    println(io, "The target's benchmark definitions and seeded random inputs are used for both commits. ",
            "PR/branch builds use the exact merge base with the fetched target/default branch. ",
            "Default-branch builds use the first parent. A cached baseline is reused only ",
            "for the same commit, GPU UUID, host, Julia/CUDA versions, resolved dependencies, ",
            "benchmark harness and parameters.\n")
    println(io, "GPU UUID: `", t_env["gpu_uuid"], "`; host: `", t_env["host"],
            "`; harness SHA-256: `", t_env["harness"], "`. ",
            "Raw samples and environment details are attached in baseline.json and target.json, alongside this report.")
end

function comparison_commits(repo)
    git(args...) = Cmd(String["git", "-C", repo, args...])
    head = readchomp(git("rev-parse", "HEAD"))
    default = get(ENV, "BUILDKITE_PIPELINE_DEFAULT_BRANCH", "main")
    branch = get(ENV, "BUILDKITE_BRANCH", readchomp(git("branch", "--show-current")))
    is_pr = get(ENV, "BUILDKITE_PULL_REQUEST", "false") ∉ ("false", "")
    if readchomp(git("rev-parse", "--is-shallow-repository")) == "true"
        run(git("fetch", "--unshallow", "origin"))
    end
    if branch == default && !is_pr
        baseline = readchomp(git("rev-parse", "$(head)^1"))
    else
        base_branch = get(ENV, "BUILDKITE_PULL_REQUEST_BASE_BRANCH", default)
        run(git("fetch", "--no-tags", "origin", base_branch))
        baseline = readchomp(git("merge-base", head, "FETCH_HEAD"))
    end
    return head, baseline
end

# Buildkite instantiates the target project; the baseline uses the same resolution.
function compare()
    repo = dirname(@__DIR__)
    git(args...) = Cmd(String["git", "-C", repo, args...])
    head, baseline = comparison_commits(repo)
    println("Baseline: ", baseline, "\nTarget: ", head)
    output = joinpath(@__DIR__, "results")
    mkpath(output)
    for file in ("baseline.json", "target.json", "report.md")
        rm(joinpath(output, file); force = true)
    end
    mktempdir() do work
        source, env = joinpath(work, "baseline"), joinpath(work, "environment")
        run(git("worktree", "add", "--detach", source, baseline))
        try
            mkpath(env)
            for file in ("Project.toml", "Manifest.toml")
                cp(joinpath(@__DIR__, file), joinpath(env, file))
            end
            withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
                Pkg.activate(env)
                Pkg.develop(PackageSpec(path = source); preserve = Pkg.PRESERVE_ALL)
                Pkg.instantiate()
                Pkg.status(; mode = Pkg.PKGMODE_MANIFEST)
            end
            # The target is always measured; only an exactly matching baseline can be reused.
            script = joinpath(@__DIR__, "runbenchmarks.jl")
            run(`$(Base.julia_cmd()) --threads=1 --project=$(@__DIR__) $script $(joinpath(output, "target.json")) $head`)
            run(`$(Base.julia_cmd()) --threads=1 --project=$env $script $(joinpath(output, "baseline.json")) $baseline --reuse`)
            report = joinpath(output, "report.md")
            text = sprint() do io
                write_report(io, read_result(joinpath(output, "baseline.json")),
                             read_result(joinpath(output, "target.json")))
            end
            write(report, text)
            print(text)
            if haskey(ENV, "BUILDKITE")
                run(pipeline(`buildkite-agent annotate --context gpu-rhs-benchmarks --style info`; stdin = report))
            end
        finally
            Pkg.activate(@__DIR__)
            run(ignorestatus(git("worktree", "remove", "--force", source)))
        end
    end
end

main() = isempty(ARGS) ? compare() : measure()

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
