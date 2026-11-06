using TrixiCUDA, CUDA, BenchmarkTools
using Dates, JSON, Pkg, Printf, SHA, TOML
include("cases.jl")

# Version of the result JSON for external readers. Cache matching does not need it: the
# harness hash covers this file, so any format change already invalidates old entries.
const RESULT_SCHEMA = 4
const TRIXICUDA_UUID = "5056661c-942a-472c-b3bc-01507f717a60"

git(args...) = Cmd(String["git", "-C", dirname(@__DIR__), args...])

# Everything besides the commit that can change the timings; baseline and target must
# match here, and a cached baseline is reused only for an identical context.
function measurement_context(result)
    env = result["environment"]
    fields = ("julia", "julia_flags", "gpu", "gpu_uuid", "host", "runtime", "driver", "threads", "harness")
    context = Dict{String, Any}(field => env[field] for field in fields)
    context["packages"] = filter(p -> first(p) != TRIXICUDA_UUID, env["packages"])
    context["parameters"] = result["parameters"]
    return context
end

function validate_result(result)
    isempty(result["benchmarks"]) && error("No benchmark results")
    for (name, case) in result["benchmarks"]
        times = case["samples_ns"]
        length(times) == case["samples"] > 0 || error("Invalid sample count: $name")
        all(t -> isfinite(t) && t > 0, times) || error("Invalid samples: $name")
        minimum(times) == case["minimum_ns"] || error("Minimum does not match samples: $name")
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
    # The identical harness fixes the initial time and DOFs per field for a given shape.
    for name in keys(t), field in ("shape", "input_hash")
        b[name][field] == t[name][field] || error("Input mismatch in $name: $field")
    end
    return nothing
end

# Write through a temporary file so an interrupted job never leaves a partial cache entry;
# `mktemp` removes the temporary file at exit if it was never moved into place.
function write_result(path, result)
    mkpath(dirname(path))
    temp, io = mktemp(dirname(path))
    JSON.json(io, result)
    close(io)
    mv(temp, path; force = true)
end

read_result(path) = JSON.parsefile(path; dicttype = Dict{String, Any})

# The key covers the commit and the full measurement context, so a cache hit matches both.
function cache_path(directory, result)
    identity = measurement_context(result)
    identity["commit"] = result["environment"]["commit"]
    key = bytes2hex(sha256(sprint(io -> TOML.print(io, identity; sorted = true))))
    return joinpath(directory, key * ".json")
end

function restore_baseline(path)
    isfile(path) || return nothing
    try
        result = read_result(path)
        validate_result(result)
        result["reused"] = true
        return result
    catch exception
        @warn "Ignoring invalid baseline cache" path exception
        return nothing
    end
end

function benchmark_rhs(case; seconds, samples)
    semi = make_problem(case)
    (; mesh, equations, boundary_conditions, source_terms, solver, cache_gpu, cache_cpu) = semi
    t = 0.0
    u_host = initial_input(semi, t)
    all(isfinite, u_host) || error("Non-finite initial state in $(case.name)")
    u = CuArray(u_host)
    du = CUDA.zeros(Float64, size(u))

    # Compile and populate caches before measuring. Fail on invalid initialized states
    # instead of publishing plausible timings for NaN-producing computations.
    CUDA.@sync TrixiCUDA.rhs_hyperbolic!(du, u, t, mesh, equations, boundary_conditions,
                                         source_terms, solver, cache_gpu, cache_cpu)
    all(isfinite, Array(du)) || error("Non-finite RHS in $(case.name)")

    # Synchronize the reset outside the timed expression. evals=1 guarantees each
    # measurement starts with du=0, including RHS implementations that accumulate.
    # The block keeps BenchmarkTools options outside CUDA.@sync.
    trial = @benchmark begin
        CUDA.@sync TrixiCUDA.rhs_hyperbolic!($du, $u, $t, $mesh, $equations,
                                             $boundary_conditions, $source_terms,
                                             $solver, $cache_gpu, $cache_cpu)
    end setup=(CUDA.@sync fill!($du, 0.0)) evals=1 seconds=seconds samples=samples
    return Dict("minimum_ns" => minimum(trial).time,
                "samples_ns" => trial.times,
                "samples" => length(trial),
                "dofs_per_field" => length(u) ÷ Trixi.nvariables(equations),
                "shape" => collect(size(u)), "initial_time" => t,
                "input_hash" => bytes2hex(sha256(reinterpret(UInt8, vec(u_host)))))
end

# Usage: runbenchmarks.jl OUTPUT.json COMMIT_SHA [--reuse]
function measure()
    path, commit = ARGS
    reuse = get(ARGS, 3, "") == "--reuse"
    CUDA.functional() || error("A functional CUDA GPU is required for RHS benchmarks")
    seconds = parse(Float64, get(ENV, "BENCHMARK_SECONDS", "1"))
    samples = parse(Int, get(ENV, "BENCHMARK_SAMPLES", "100"))
    case_filter = get(ENV, "BENCHMARK_FILTER", "")
    cases = [case for case in RHS_CASES if occursin(case_filter, case.name)]
    isempty(cases) && error("No benchmark cases match '$case_filter'")
    packages = Dict(string(id) => Dict("name" => info.name, "version" => string(info.version),
                                       "tree" => string(info.tree_hash))
                    for (id, info) in Pkg.dependencies() if info.version !== nothing)
    # Include all code and project settings that can change the measured workload.
    harness_files = vcat(["cases.jl", "runbenchmarks.jl", "Project.toml"],
                         relpath.(RHS_CASE_FILES, @__DIR__))
    files = Dict(file => read(joinpath(@__DIR__, file), String) for file in harness_files)
    harness = bytes2hex(sha256(sprint(io -> TOML.print(io, files; sorted = true))))
    # Command-line code generation flags are not part of the harness files or VERSION.
    flags = Base.JLOptions()
    julia_flags = Dict("opt_level" => Int(flags.opt_level), "check_bounds" => Int(flags.check_bounds))
    environment = Dict("commit" => commit, "julia" => string(VERSION), "julia_flags" => julia_flags,
                       "gpu" => CUDA.name(CUDA.device()), "gpu_uuid" => string(CUDA.uuid(CUDA.device())),
                       "host" => get(ENV, "RUNNER_NAME", "unknown-runner"), "harness" => harness,
                       "runtime" => string(CUDA.runtime_version()),
                       "driver" => CUDA.has_nvml() ? string(CUDA.NVML.driver_version()) : string(CUDA.driver_version()),
                       "threads" => Threads.nthreads(), "packages" => packages,
                       "cuda_versioninfo" => sprint(CUDA.versioninfo),
                       "measured_at" => string(now(UTC)),
                       "build_url" => get(ENV, "BENCHMARK_RUN_URL", "local"))
    output = Dict("schema" => RESULT_SCHEMA, "environment" => environment,
                  "parameters" => Dict("seconds" => seconds, "samples" => samples,
                       "precision" => "Float64", "evals" => 1,
                       "initialization" => "classical initial condition", "initial_time" => 0.0,
                       "filter" => case_filter), "reused" => false)
    # Keep the cache in this runner's depot scratchspace.
    directory = get(ENV, "BENCHMARK_CACHE",
                    joinpath(first(DEPOT_PATH), "scratchspaces", TRIXICUDA_UUID, "rhs-benchmarks"))
    cached_file = cache_path(directory, output)
    if reuse
        cached = restore_baseline(cached_file)
        if cached !== nothing
            measured_at = cached["environment"]["measured_at"]
            @info "Matching baseline measurement found; reusing it" commit measured_at
            write_result(path, cached)
            return nothing
        end
        @info "No matching baseline measurement; benchmarking" commit
    end
    results = Dict{String, Any}()
    for case in cases
        @info "Benchmarking $(case.name)"
        results[case.name] = benchmark_rhs(case; seconds, samples)
        # Avoid retaining all case allocations on small benchmark GPUs.
        GC.gc(true)
        CUDA.reclaim()
    end
    output["benchmarks"] = results
    write_result(path, output)
    write_result(cached_file, output)
    return nothing
end

function write_report(io, baseline, target)
    validate_comparison(baseline, target)
    b_env, t_env = baseline["environment"], target["environment"]
    b, t = baseline["benchmarks"], target["benchmarks"]
    parameters = target["parameters"]
    speedup = exp(sum(log(b[name]["minimum_ns"] / t[name]["minimum_ns"]) for name in keys(t)) / length(t))
    println(io, "## GPU RHS Benchmarks\n")
    println(io, "Baseline: `", b_env["commit"], "`  \nTarget: `", t_env["commit"], "`\n")
    gpu = replace(t_env["gpu"], r"(?i)\s+laptop gpu$" => "")
    println(io, "**Environment:** ", gpu, " with Julia ", t_env["julia"],
            ", CUDA runtime ", t_env["runtime"], ", and NVIDIA driver ", t_env["driver"], ".\n")
    @printf(io, "**Results:** %d cases; geometric mean speedup **%.3fx**\n\n", length(t), speedup)
    println(io, "| RHS case | DOFs/field | Samples (base/target) | ",
            "Baseline min (ms) | Target min (ms) | Speedup |")
    println(io, "|:--|--:|--:|--:|--:|--:|")
    for name in sort!(collect(keys(t)))
        base, head = b[name], t[name]
        bt, tt = base["minimum_ns"], head["minimum_ns"]
        @printf(io, "| %s | %d | %d / %d | %.4f | %.4f | %.3fx |\n",
                name, head["dofs_per_field"], base["samples"], head["samples"],
                bt / 1e6, tt / 1e6, bt / tt)
    end
    println(io, "\nSpeedup > 1 indicates improvement. ",
            "Informational only: no regression threshold is applied. ",
            "Small differences may be measurement noise.\n")
    for (label, result) in (("Baseline", baseline), ("Target", target))
        env = result["environment"]
        println(io, label, " measurement: ", result["reused"] ? "reused" : "fresh",
                "; ", env["measured_at"], " UTC; ", env["build_url"], ".  ")
    end
    # Keep the numbers first; the settings stay one click away.
    println(io, "\n<details>\n<summary><b>Benchmark settings</b></summary>\n\n",
            "**Workload:** The benchmark measures `TrixiCUDA.rhs_hyperbolic!` using Float64 arrays ",
            "and one Julia thread. Each RHS call is GPU-synchronized, and the minimum wall time ",
            "is reported. Both commits use the target's case definitions, with classical ",
            "initial conditions, polynomial degrees, meshes, and fluxes adapted from ",
            "Trixi/TrixiShallowWater examples. Inputs are initialized at t = 0.\n\n",
            "**Sampling:** After one warm-up RHS call per case, each sample measures one ",
            "synchronized RHS call. Sampling stops after ", parameters["samples"],
            " samples or a ", parameters["seconds"], " s sampling budget, whichever comes ",
            "first, to limit benchmark runtime. Sample counts are shown as baseline / target.\n\n",
            "**Target:** Push runs measure that push's head commit. Manual runs select the ",
            "specified PR's current head, or the selected default-branch commit when no PR ",
            "is specified.\n\n",
            "**Baseline:** PR runs use the exact merge base of the target commit and ",
            "the fetched PR target branch (or the default branch for local branch runs). ",
            "Default-branch runs use the target commit's first parent.\n\n",
            "**Cache reuse:** The target is always measured. A valid stored result can be ",
            "reused for the baseline only when its commit equals the selected baseline commit ",
            "and the GPU UUID/runner, Julia/CUDA/driver versions, dependencies, harness, and ",
            "parameters all match. Otherwise, the baseline is measured again.\n\n</details>\n")
end

# Default-branch runs compare with the first parent; PR and other branch runs compare
# with the merge base of the target and the fetched base branch.
function comparison_commits()
    head = readchomp(git("rev-parse", "HEAD"))
    if readchomp(git("rev-parse", "--is-shallow-repository")) == "true"
        run(git("fetch", "--unshallow", "--no-tags", "origin", head))
    end
    default = get(ENV, "BENCHMARK_DEFAULT_BRANCH", "main")
    branch = get(() -> readchomp(git("branch", "--show-current")), ENV, "GITHUB_REF_NAME")
    if branch == default && isempty(get(ENV, "BENCHMARK_PR", ""))
        return head, readchomp(git("rev-parse", "$(head)^1"))
    end
    run(git("fetch", "--no-tags", "origin", get(ENV, "BENCHMARK_BASE_BRANCH", default)))
    return head, readchomp(git("merge-base", head, "FETCH_HEAD"))
end

# The workflow instantiates the target project; the baseline uses the same resolution.
function compare()
    head, baseline = comparison_commits()
    if haskey(ENV, "GITHUB_OUTPUT")
        open(io -> println(io, "baseline=", baseline), ENV["GITHUB_OUTPUT"], "a")
    end
    output = joinpath(@__DIR__, "results")
    rm(output; force = true, recursive = true)
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
            # Save the exact resolved projects separately from the three report files.
            for (label, project) in (("target", @__DIR__), ("baseline", env))
                mkpath(joinpath(output, "environments", label))
                for file in ("Project.toml", "Manifest.toml")
                    cp(joinpath(project, file), joinpath(output, "environments", label, file))
                end
            end
            # The target is always measured; only an exactly matching baseline can be reused.
            # One Julia thread keeps the CPU-side parts (e.g. shock capturing) reproducible.
            script = joinpath(@__DIR__, "runbenchmarks.jl")
            @info "Target measurement" commit = head
            run(`$(Base.julia_cmd()) --threads=1 --project=$(@__DIR__) $script $(joinpath(output, "target.json")) $head`)
            @info "Baseline measurement" commit = baseline
            run(`$(Base.julia_cmd()) --threads=1 --project=$env $script $(joinpath(output, "baseline.json")) $baseline --reuse`)
            text = sprint() do io
                write_report(io, read_result(joinpath(output, "baseline.json")),
                             read_result(joinpath(output, "target.json")))
            end
            report = joinpath(output, "report.md")
            write(report, text)
            @info "Benchmark report saved" report
            if haskey(ENV, "GITHUB_STEP_SUMMARY")
                open(io -> write(io, text), ENV["GITHUB_STEP_SUMMARY"], "a")
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
