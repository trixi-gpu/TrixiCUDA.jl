using TrixiCUDA, CUDA, BenchmarkTools
using Dates, JSON, Pkg, Printf, SHA
include("cases.jl")

git(args...) = Cmd(String["git", "-C", dirname(@__DIR__), args...])

function validate_comparison(baseline, target)
    b, t = baseline["benchmarks"], target["benchmarks"]
    keys(b) == keys(t) || error("Benchmark case sets differ")
    # The case definitions fix the initial time, and the shape fixes the DOFs per field.
    for name in keys(t), field in ("shape", "input_hash")
        b[name][field] == t[name][field] || error("Input mismatch in $name: $field")
    end
    return nothing
end

read_result(path) = JSON.parsefile(path; dicttype = Dict{String, Any})

function benchmark_rhs(case; t, seconds, samples)
    semi = make_problem(case)
    (; mesh, equations, boundary_conditions, source_terms, solver, cache_gpu, cache_cpu) = semi
    u_host = initial_input(semi, t)
    all(isfinite, u_host) || error("Non-finite initial state in $(case.name)")
    u = CuArray(u_host)
    du = CUDA.zeros(Float64, size(u))

    # Warm up the RHS before timing to trigger compilation and initialization.
    # Check that the warm-up output is finite, since NaN or Inf can trigger
    # different execution paths and make timings harder to interpret.
    CUDA.@sync TrixiCUDA.rhs_hyperbolic!(du, u, t, mesh, equations, boundary_conditions,
                                         source_terms, solver, cache_gpu, cache_cpu)
    all(isfinite, Array(du)) || error("Non-finite RHS in $(case.name)")

    # Time the synchronized RHS call. Reset du to zero before each sample so 
    # every timed call starts with the same output buffer contents. Wait for 
    # the reset to finish before starting the timer.
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

function measure()
    path, commit = ARGS
    CUDA.functional() || error("A functional CUDA GPU is required for RHS benchmarks")
    # Initial time for every case; the results and report record this same value.
    t = 0.0
    seconds = parse(Float64, get(ENV, "BENCHMARK_SECONDS", "1"))
    samples = parse(Int, get(ENV, "BENCHMARK_SAMPLES", "100"))
    case_filter = get(ENV, "BENCHMARK_FILTER", "")
    cases = [case for case in RHS_CASES if occursin(case_filter, case.name)]
    isempty(cases) && error("No benchmark cases match '$case_filter'")
    packages = Dict(string(id) => Dict("name" => info.name, "version" => string(info.version),
                                       "tree" => string(info.tree_hash))
                    for (id, info) in Pkg.dependencies() if info.version !== nothing)
    environment = Dict("commit" => commit, "julia" => string(VERSION),
                       "gpu" => CUDA.name(CUDA.device()), "gpu_uuid" => string(CUDA.uuid(CUDA.device())),
                       "host" => get(ENV, "RUNNER_NAME", "unknown-runner"),
                       "runtime" => string(CUDA.runtime_version()),
                       "driver" => CUDA.has_nvml() ? string(CUDA.NVML.driver_version()) : string(CUDA.driver_version()),
                       "threads" => Threads.nthreads(), "packages" => packages,
                       "cuda_versioninfo" => sprint(CUDA.versioninfo),
                       "measured_at" => string(now(UTC)),
                       "build_url" => get(ENV, "BENCHMARK_RUN_URL", "local"))
    output = Dict("environment" => environment,
                  "parameters" => Dict("seconds" => seconds, "samples" => samples,
                       "precision" => "Float64", "evals" => 1,
                       "initialization" => "classical initial condition", "initial_time" => t,
                       "filter" => case_filter))
    results = Dict{String, Any}()
    for case in cases
        @info "Benchmarking $(case.name)"
        results[case.name] = benchmark_rhs(case; t, seconds, samples)
        # Avoid retaining all case allocations on small benchmark GPUs.
        GC.gc(true)
        CUDA.reclaim()
    end
    output["benchmarks"] = results
    mkpath(dirname(path))
    write(path, JSON.json(output))
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
        measured_at = Dates.format(DateTime(env["measured_at"]), dateformat"yyyy-mm-dd HH:MM")
        println(io, label, " measurement: ", measured_at, " UTC; ", env["build_url"], ".  ")
    end
    # Keep the numbers first; the settings stay one click away.
    println(io, "\n<details>\n<summary><b>Benchmark settings</b></summary>\n\n",
            "<br>\n\n",
            "**Workload:** The benchmark measures `TrixiCUDA.rhs_hyperbolic!` using Float64 arrays ",
            "and one Julia thread. Each RHS call is GPU-synchronized, and the minimum wall time ",
            "is reported. Both commits use the target's case definitions, with classical ",
            "initial conditions, polynomial degrees, meshes, and fluxes adapted from ",
            "Trixi/TrixiShallowWater examples. Inputs are initialized at t = ",
            @sprintf("%g", parameters["initial_time"]), ".\n\n",
            "**Sampling:** After one warm-up RHS call per case, each sample measures one ",
            "synchronized RHS call. Sampling stops after ", parameters["samples"],
            " samples or a ", parameters["seconds"], " s sampling budget, whichever comes ",
            "first, to limit benchmark runtime. Sample counts are shown as baseline / target.\n\n",
            "**Target:** Push runs measure that push's head commit. Manual runs measure the ",
            "latest commit of the selected branch: the default branch without a PR number, ",
            "or the head branch of the specified PR.\n\n",
            "**Baseline:** PR runs use the exact merge base of the target commit and ",
            "the fetched PR target branch (or the default branch for local branch runs). ",
            "Default-branch runs use the target commit's first parent.\n\n</details>\n")
end

function comparison_commits()
    head = readchomp(git("rev-parse", "HEAD"))
    default = get(ENV, "BENCHMARK_DEFAULT_BRANCH", "main")
    branch = get(() -> readchomp(git("branch", "--show-current")), ENV, "GITHUB_REF_NAME")
    if branch == default && isempty(get(ENV, "BENCHMARK_PR", ""))
        return head, readchomp(git("rev-parse", "$(head)^1"))
    end
    run(git("fetch", "--no-tags", "origin", get(ENV, "BENCHMARK_BASE_BRANCH", default)))
    return head, readchomp(git("merge-base", head, "FETCH_HEAD"))
end

# Both runs use the target's benchmark project. Each run uses its own revision
# of source code and resolves its dependencies independently from scratch.
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
            cp(joinpath(@__DIR__, "Project.toml"), joinpath(env, "Project.toml"))
            withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
                Pkg.activate(env)
                Pkg.develop(PackageSpec(path = source))
                Pkg.instantiate()
                Pkg.status(; mode = Pkg.PKGMODE_MANIFEST)
            end
            for (label, project) in (("target", @__DIR__), ("baseline", env))
                mkpath(joinpath(output, "environments", label))
                for file in ("Project.toml", "Manifest.toml")
                    cp(joinpath(project, file), joinpath(output, "environments", label, file))
                end
            end
            # One Julia thread keeps the CPU-side parts (e.g. shock capturing) reproducible.
            script = joinpath(@__DIR__, "runbenchmarks.jl")
            @info "Target measurement" commit = head
            run(`$(Base.julia_cmd()) --threads=1 --project=$(@__DIR__) $script $(joinpath(output, "target.json")) $head`)
            @info "Baseline measurement" commit = baseline
            run(`$(Base.julia_cmd()) --threads=1 --project=$env $script $(joinpath(output, "baseline.json")) $baseline`)
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
