using TrixiCUDA, CUDA, BenchmarkTools
using Dates, JSON, Pkg, Printf, Random, SHA
include("cases.jl")

const NOISE_BAND = 0.03  # calibrated from runs that compare identical code (#270)
const RESAMPLES = 9_999
const PROCESSES = 3  # per revision, run alternately: baseline, target, baseline, ...

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

function benchmark_rhs(case; t, samples)
    semi = make_problem(case)
    (; equations) = semi
    u_host = initial_input(semi, t)
    all(isfinite, u_host) || error("Non-finite initial state in $(case.name)")
    u = CuArray(u_host)
    du = zero(u)

    # Warm up the RHS before timing to trigger compilation and initialization.
    # Check that the warm-up output is finite, since NaN or Inf can trigger
    # different execution paths and make timings harder to interpret.
    CUDA.@sync TrixiCUDA.rhs_hyperbolic!(du, u, semi, t)
    all(isfinite, Array(du)) || error("Non-finite RHS in $(case.name)")

    # Time the synchronized RHS call. Reset du to zero before each sample so 
    # every timed call starts with the same output buffer contents. Wait for 
    # the reset to finish before starting the timer.
    # BenchmarkTools needs a finite time budget; this one never stops sampling early.
    trial = @benchmark begin
        CUDA.@sync TrixiCUDA.rhs_hyperbolic!($du, $u, $semi, $t)
    end setup=(CUDA.@sync fill!($du, 0)) evals=1 seconds=1e6 samples=samples
    return Dict("minimum_ns" => minimum(trial).time,
                "samples_ns" => trial.times,
                "samples" => length(trial),
                "dofs_per_field" => div(length(u), Trixi.nvariables(equations)),
                "shape" => collect(size(u)), "initial_time" => t,
                "input_hash" => bytes2hex(sha256(reinterpret(UInt8, vec(u_host)))))
end

function measure()
    path, commit = ARGS
    CUDA.functional() || error("A functional CUDA GPU is required for RHS benchmarks")
    # Initial time for every case; the results and report record this same value.
    t = 0.0
    samples = 100
    packages = Dict(string(id) => Dict("name" => info.name, "version" => string(info.version),
                                       "tree" => string(info.tree_hash))
                    for (id, info) in Pkg.dependencies() if info.version !== nothing)
    environment = Dict("commit" => commit, "julia" => string(VERSION),
                       "gpu" => CUDA.name(CUDA.device()), "gpu_uuid" => string(CUDA.uuid(CUDA.device())),
                       "host" => ENV["RUNNER_NAME"],
                       "runtime" => string(CUDA.runtime_version()),
                       "driver" => CUDA.has_nvml() ? string(CUDA.NVML.driver_version()) : string(CUDA.driver_version()),
                       "threads" => Threads.nthreads(), "packages" => packages,
                       "cuda_versioninfo" => sprint(CUDA.versioninfo),
                       "measured_at" => string(now(UTC)),
                       "build_url" => ENV["BENCHMARK_RUN_URL"])
    output = Dict("environment" => environment,
                  "parameters" => Dict("samples" => samples,
                       "precision" => "Float64", "evals" => 1,
                       "initialization" => "classical initial condition", "initial_time" => t))
    results = Dict{String, Any}()
    for case in RHS_CASES
        @info "Benchmarking $(case.name)"
        results[case.name] = benchmark_rhs(case; t, samples)
        # Avoid retaining all case allocations on small benchmark GPUs.
        GC.gc(true)
        CUDA.reclaim()
    end
    output["benchmarks"] = results
    mkpath(dirname(path))
    write(path, JSON.json(output))
    return nothing
end

median!(x) = (n = length(x); isodd(n) ? partialsort!(x, (n + 1) ÷ 2) : sum(partialsort!(x, (n ÷ 2):(n ÷ 2 + 1))) / 2)

# Median of all pairwise ratios between baseline and target process medians.
median_ratio(base, target) = median!(vec([b / t for b in base, t in target]))

# Two-level resampling: draw processes with replacement, then samples within each process.
resampled_medians(rng, processes) = [median!(rand(rng, p, length(p))) for p in rand(rng, processes, length(processes))]

# 95% percentile bootstrap interval. With 9,999 resamples the interval ends are
# exactly the 250th smallest and the 250th largest estimates.
function speedup_interval(base, target, rng)
    estimates = sort!([median_ratio(resampled_medians(rng, base), resampled_medians(rng, target))
                       for _ in 1:RESAMPLES])
    k = round(Int, 0.025 * (RESAMPLES + 1))
    return estimates[k], estimates[end + 1 - k]
end

# The seed depends on the case and both commits, so a report is reproducible from its raw samples.
bootstrap_rng(name, baseline, target) = Xoshiro(first(reinterpret(UInt64, sha256(name * baseline * target)[1:8])))

function verdict((low, high))
    low > 1 + NOISE_BAND && return "faster"
    high < 1 / (1 + NOISE_BAND) && return "slower"
    1 / (1 + NOISE_BAND) <= low && high <= 1 + NOISE_BAND && return "no change"
    return "inconclusive"
end

samples_by_process(results, name) = [Float64.(r["benchmarks"][name]["samples_ns"]) for r in results]

function compare_case(name, baselines, targets, baseline_commit, target_commit)
    b, t = samples_by_process(baselines, name), samples_by_process(targets, name)
    interval = speedup_interval(b, t, bootstrap_rng(name, baseline_commit, target_commit))
    return (; name, dofs = targets[1]["benchmarks"][name]["dofs_per_field"],
            base_min = minimum(minimum, b), target_min = minimum(minimum, t),
            speedup = median_ratio(median!.(copy.(b)), median!.(copy.(t))), interval,
            verdict = verdict(interval))
end

function write_report(io, baselines, targets)
    foreach(r -> validate_comparison(r, targets[1]), [baselines; targets[2:end]])
    baseline, target = baselines[1], targets[1]
    b_env, t_env = baseline["environment"], target["environment"]
    parameters = target["parameters"]
    rows = [compare_case(name, baselines, targets, b_env["commit"], t_env["commit"])
            for name in sort!(collect(keys(target["benchmarks"])))]
    speedup = exp(sum(row -> log(row.speedup), rows) / length(rows))
    counts = join(("$(count(row -> row.verdict == v, rows)) $v"
                   for v in ("faster", "slower", "no change", "inconclusive")), ", ")
    band = @sprintf("%.3f–%.3f", 1 / (1 + NOISE_BAND), 1 + NOISE_BAND)

    println(io, "## GPU RHS Benchmarks\n")
    println(io, "Baseline: `", b_env["commit"], "`  \nTarget: `", t_env["commit"], "`\n")
    gpu = replace(t_env["gpu"], r"(?i)\s+laptop gpu$" => "")
    println(io, "**Environment:** ", gpu, " with Julia ", t_env["julia"],
            ", CUDA runtime ", t_env["runtime"], ", and NVIDIA driver ", t_env["driver"], ".\n")
    @printf(io, "**Results:** %d cases; geometric mean speedup **%.3fx** (%s)\n\n",
            length(rows), speedup, counts)
    println(io, "| RHS case | DOFs/field | Baseline min (ms) | Target min (ms) | ",
            "Speedup | 95% interval | Verdict |")
    println(io, "|:--|--:|--:|--:|--:|:-:|:--|")
    for row in rows
        @printf(io, "| %s | %d | %.4f | %.4f | %.3fx | %.3f–%.3f | %s |\n",
                row.name, row.dofs, row.base_min / 1e6, row.target_min / 1e6, row.speedup,
                row.interval[1], row.interval[2], row.verdict)
    end
    println(io, "\nSpeedup > 1 means the target is faster. A case is faster or slower only when ",
            "its whole 95% interval lies outside the ±", round(Int, 100 * NOISE_BAND),
            "% noise band (", band, "), no change when the interval lies inside it, ",
            "and inconclusive otherwise.\n")
    for (label, result) in (("Baseline", baseline), ("Target", target))
        env = result["environment"]
        measured_at = Dates.format(DateTime(env["measured_at"]), dateformat"yyyy-mm-dd HH:MM")
        println(io, label, " measurement: ", measured_at, " UTC; ", env["build_url"], ".  ")
    end
    # Keep the numbers first; the settings stay one click away.
    println(io, "\n<details>\n<summary><b>Benchmark settings</b></summary>\n\n",
            "<br>\n\n",
            "**Workload:** The benchmark measures `TrixiCUDA.rhs_hyperbolic!` using Float64 arrays ",
            "and one Julia thread. The cases use classical initial conditions, polynomial degrees, ",
            "meshes, and fluxes adapted from Trixi/TrixiShallowWater examples. ",
            "Inputs are initialized at t = ", @sprintf("%g", parameters["initial_time"]), ".\n\n",
            "**Comparison setup:** Both revisions are measured on the same GPU using the ",
            "target commit's benchmark harness, with identical cases and timing settings, in ",
            PROCESSES, " separate Julia processes per revision that alternate between baseline and ",
            "target. Each revision loads the source from its respective commit and resolves ",
            "its Julia dependencies independently. The comparison ",
            "captures the combined performance impact of source code and dependency changes.\n\n",
            "**Sampling:** After one warm-up RHS call per case, each process records ",
            parameters["samples"], " samples per case, each timing one GPU-synchronized RHS call.\n\n",
            "**Statistics:** The speedup is the median of all ratios between baseline and target ",
            "process medians. Its 95% interval is a two-level percentile bootstrap with ", RESAMPLES,
            " resamples that draws processes, then samples within each process; ",
            "the random seed is derived from the case name and both commits, so the same samples ",
            "always give the same interval. Min columns show the fastest sample of each revision. ",
            "The noise band was calibrated with runs that compare identical code.\n\n",
            "**Target:** The head commit of a push whose title contains `[benchmark mini]`.\n\n",
            "**Baseline:** PR runs use the exact merge base of the target commit and ",
            "the fetched PR base branch. Default-branch runs use the target commit's first parent.\n\n</details>\n")
end

function comparison_commits()
    head = readchomp(git("rev-parse", "HEAD"))
    if ENV["GITHUB_REF_NAME"] == ENV["BENCHMARK_DEFAULT_BRANCH"]
        return head, readchomp(git("rev-parse", "$(head)^1"))
    end
    run(git("fetch", "--no-tags", "origin", ENV["BENCHMARK_BASE_BRANCH"]))
    return head, readchomp(git("merge-base", head, "FETCH_HEAD"))
end

# Both runs use the target's benchmark project. Each run uses its own revision
# of source code and resolves its dependencies independently from scratch.
function compare()
    head, baseline = comparison_commits()
    open(io -> println(io, "baseline=", baseline), ENV["GITHUB_OUTPUT"], "a")
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
            result(label, i) = joinpath(output, "$label-$i.json")
            for i in 1:PROCESSES, (label, project, commit) in (("baseline", env, baseline), ("target", @__DIR__, head))
                @info "Measurement $i of $PROCESSES" label commit
                run(`$(Base.julia_cmd()) --threads=1 --project=$project $script $(result(label, i)) $commit`)
            end
            results = Dict(label => [read_result(result(label, i)) for i in 1:PROCESSES]
                           for label in ("baseline", "target"))
            text = sprint(io -> write_report(io, results["baseline"], results["target"]))
            report = joinpath(output, "report.md")
            write(report, text)
            open(io -> write(io, text), ENV["GITHUB_STEP_SUMMARY"], "a")
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
