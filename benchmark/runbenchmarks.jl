using TrixiCUDA, CUDA, BenchmarkTools
using Dates, JSON, Pkg, Printf, Random, SHA
include("cases.jl")
include("sweeps.jl")

const NOISE_BAND = 0.03
const RESAMPLES = 9_999
const PROCESSES = 3
const VERDICTS = ("faster", "slower", "no change", "inconclusive")

git(args...) = Cmd(String["git", "-C", dirname(@__DIR__), args...])

function validate_comparison(baseline, target)
    b, t = baseline["benchmarks"], target["benchmarks"]
    keys(b) == keys(t) || error("Benchmark case sets differ")
    for name in keys(t), field in ("shape", "input_hash")
        (haskey(b[name], "error") || haskey(t[name], "error")) && continue
        b[name][field] == t[name][field] || error("Input mismatch in $name: $field")
    end
    return nothing
end

read_result(path) = JSON.parsefile(path; dicttype = Dict{String, Any})

function benchmark_rhs(case; t, samples, kwargs...)
    semi = make_problem(case; kwargs...)
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
    trial = @benchmark begin
        CUDA.@sync TrixiCUDA.rhs_hyperbolic!($du, $u, $semi, $t)
    end setup=(CUDA.@sync fill!($du, 0)) evals=1 seconds=1e6 samples=samples
    return Dict("samples_ns" => trial.times,
                "dofs_per_field" => div(length(u), Trixi.nvariables(equations)),
                "shape" => collect(size(u)), "initial_time" => t,
                "input_hash" => bytes2hex(sha256(reinterpret(UInt8, vec(u_host)))))
end

point_name(name, point) = "$name p$(point.polydeg) L$(point.level)"

function benchmark_points(scope)
    scope == "mini" && return [(case.name, case, (;), (;)) for case in RHS_CASES]
    return [(point_name(case.name, point), case,
             (; point.polydeg, initial_refinement_level = point.level),
             (; case = case.name, point.polydeg, point.level, point.sweeps))
            for case in RHS_CASES for point in sweep_points(case.name)]
end

function measure()
    path, commit, scope = ARGS
    CUDA.functional() || error("A functional CUDA GPU is required for RHS benchmarks")
    # Initial time for every case; the results and report record this same value.
    t = 0.0
    # The sweep takes fewer samples per point than mini, so that sweep can finish 
    # in a few hours. Revisit 50 once the RHS gets faster and 100 becomes affordable.
    samples = scope == "mini" ? 100 : 50
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
                  "parameters" => Dict("scope" => scope,
                       "samples" => samples,
                       "precision" => "Float64", "evals" => 1,
                       "initialization" => "classical initial condition", "initial_time" => t))
    results = Dict{String, Any}()
    for (name, case, kwargs, point) in benchmark_points(scope)
        @info "Benchmarking $name"
        info = Dict{String, Any}(string(key) => value for (key, value) in pairs(point))
        # A failing sweep point is recorded and the sweep goes on; mini stops at any failure.
        results[name] = try
            merge(info, benchmark_rhs(case; t, samples, kwargs...))
        catch err
            (scope == "mini" || err isa InterruptException) && rethrow()
            merge(info, Dict("error" => first(replace(sprint(showerror, err), r"\s+" => " "), 500)))
        end
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
median_ratio(base, target) = median!(vec([b / t for b in base, t in target]))
resampled_medians(rng, processes) = [median!(rand(rng, p, length(p))) for p in rand(rng, processes, length(processes))]

function speedup_interval(base, target, rng)
    estimates = sort!([median_ratio(resampled_medians(rng, base), resampled_medians(rng, target))
                       for _ in 1:RESAMPLES])
    k = round(Int, 0.025 * (RESAMPLES + 1))
    return estimates[k], estimates[end + 1 - k]
end

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

function compare_point(name, baselines, targets, baseline_commit, target_commit)
    for (label, results) in (("baseline", baselines), ("target", targets)), r in results
        message = get(r["benchmarks"][name], "error", nothing)
        message === nothing || return (; name, error = "$label: $message")
    end
    return compare_case(name, baselines, targets, baseline_commit, target_commit)
end

gpu_name(env) = replace(env["gpu"], r"(?i)\s+laptop gpu$" => "")
band_text() = @sprintf("%.3f–%.3f", 1 / (1 + NOISE_BAND), 1 + NOISE_BAND)
geomean(speedups) = isempty(speedups) ? NaN : exp(sum(log, speedups) / length(speedups))

function write_header(io, title, b_env, t_env)
    println(io, "## ", title, "\n")
    println(io, "Baseline: `", b_env["commit"], "`  \nTarget: `", t_env["commit"], "`\n")
    println(io, "**Environment:** ", gpu_name(t_env), " with Julia ", t_env["julia"],
            ", CUDA runtime ", t_env["runtime"], ", and NVIDIA driver ", t_env["driver"], ".\n")
end

function write_footer(io, unit, baseline, target)
    println(io, "\nSpeedup > 1 means the target is faster. A ", unit, " is faster or slower only when ",
            "its whole 95% interval lies outside the ±", round(Int, 100 * NOISE_BAND),
            "% noise band (", band_text(), "), no change when the interval lies inside it, ",
            "and inconclusive otherwise.\n")
    for (label, result) in (("Baseline", baseline), ("Target", target))
        env = result["environment"]
        measured_at = Dates.format(DateTime(env["measured_at"]), dateformat"yyyy-mm-dd HH:MM")
        println(io, label, " measurement: ", measured_at, " UTC; ", env["build_url"], ".  ")
    end
    write_settings(io, target["parameters"])
end

function write_settings(io, parameters)
    mini = parameters["scope"] == "mini"
    unit = mini ? "case" : "point"
    workload = mini ?
               ("Initial conditions, polynomial degrees, meshes, and fluxes are adapted from the " *
                "corresponding Trixi.jl and TrixiShallowWater.jl examples, and the state is ") :
               ("Initial conditions, meshes, and fluxes are adapted from the corresponding Trixi.jl " *
                "and TrixiShallowWater.jl examples. The level sweep varies the refinement level L " *
                "at p = $(LEVEL_SWEEP.polydeg), and the degree sweep varies the polynomial degree p " *
                "at L = $(DEGREE_SWEEP.level[1]) (1D), $(DEGREE_SWEEP.level[2]) (2D), and " *
                "$(DEGREE_SWEEP.level[3]) (3D); a few points are left out until their issues are " *
                "fixed. The state is ")
    println(io, "\n<details>\n<summary><b>Benchmark settings</b></summary>\n\n",
            "<br>\n\n",
            "**Workload:** Each ", unit, " times `TrixiCUDA.rhs_hyperbolic!` with Float64 arrays on one ",
            "Julia thread. ", workload,
            "initialized at t = ", @sprintf("%g", parameters["initial_time"]), ".\n\n",
            "**Comparison setup:** Both revisions run on the same GPU with the target commit's ",
            "benchmark harness, so cases and timing settings are identical. Each revision loads its ",
            "own source and resolves its own Julia dependencies, so the comparison reflects source ",
            "and dependency changes together. Each revision is measured in ", PROCESSES,
            " separate Julia processes, run in alternating order (baseline, target, baseline, …), ",
            "so that variation between processes, such as a different GPU clock state, is ",
            "reflected in the interval rather than mistaken for a performance change.\n\n",
            "**Sampling:** After one warm-up call per ", unit, ", each process records ",
            parameters["samples"], " samples per ", unit, ", each timing a single GPU-synchronized ",
            "RHS call.\n\n",
            "**Statistics:** The speedup is the median of all ratios between baseline and target ",
            "process medians. Its 95% interval is a two-level percentile bootstrap with ", RESAMPLES,
            " resamples, which draws processes and then samples within each drawn process. The ",
            "random seed is derived from the ", mini ? "case name" : "point", " and both commits, ",
            "so identical samples always yield the same interval. The ±", round(Int, 100 * NOISE_BAND),
            "% noise band was calibrated with runs that compare identical code, in which no case was ",
            "reported faster or slower.",
            mini ? (" The minimum columns show the fastest sample of each revision across all " *
                    "processes.") : "", "\n\n",
            "**Target:** The head commit of a push whose commit title contains `[benchmark ",
            parameters["scope"], "]`.\n\n",
            "**Baseline:** For pull requests, the merge base of the target commit and the fetched ",
            "PR base branch; on the default branch, the first parent of the target commit.",
            "\n\n</details>\n")
end

function comparison_rows(scope, baselines, targets)
    b_commit, t_commit = (r["environment"]["commit"] for r in (baselines[1], targets[1]))
    return [merge(compare_point(name, baselines, targets, b_commit, t_commit), point)
            for (name, _, _, point) in benchmark_points(scope)]
end

function write_mini_report(io, baselines, targets, rows)
    baseline, target = baselines[1], targets[1]
    b_env, t_env = baseline["environment"], target["environment"]
    counts = join(("$(count(row -> row.verdict == v, rows)) $v" for v in VERDICTS), ", ")

    write_header(io, "GPU RHS Benchmark Mini", b_env, t_env)
    @printf(io, "**Results:** %d cases; geometric mean speedup **%.3fx** (%s)\n\n",
            length(rows), geomean([row.speedup for row in rows]), counts)
    println(io, "| RHS case | DOFs/field | Baseline min (ms) | Target min (ms) | ",
            "Speedup | 95% interval | Verdict |")
    println(io, "|:--|--:|--:|--:|--:|:-:|:--|")
    for row in rows
        @printf(io, "| %s | %d | %.4f | %.4f | %.3fx | %.3f–%.3f | %s |\n",
                row.name, row.dofs, row.base_min / 1e6, row.target_min / 1e6, row.speedup,
                row.interval[1], row.interval[2], row.verdict)
    end
    write_footer(io, "case", baseline, target)
end

function write_sweep_report(io, baselines, targets, rows)
    baseline, target = baselines[1], targets[1]
    measured = filter(row -> !haskey(row, :error), rows)
    tally(group) = join(vcat(["$(count(row -> get(row, :verdict, "") == v, group)) $v" for v in VERDICTS],
                             "$(count(row -> haskey(row, :error), group)) failed"), ", ")

    write_header(io, "GPU RHS Benchmark Sweep", baseline["environment"], target["environment"])
    @printf(io, "**Results:** %d points; geometric mean speedup **%.3fx** (%s)\n\n",
            length(rows), geomean([row.speedup for row in measured]), tally(rows))
    println(io, "| Sweep | Points | Geometric mean speedup | Verdicts |")
    println(io, "|:--|--:|--:|:--|")
    for sweep in ("level", "degree"), d in 1:3
        group = filter(row -> sweep in row.sweeps && case_ndims(row.case) == d, rows)
        fixed = sweep == "level" ? "p = $(LEVEL_SWEEP.polydeg)" : "L = $(DEGREE_SWEEP.level[d])"
        @printf(io, "| %dD %s sweep, %s | %d | %.3fx | %s |\n", d, sweep, fixed, length(group),
                geomean([row.speedup for row in group if !haskey(row, :error)]), tally(group))
    end

    changed = filter(row -> get(row, :verdict, "") in ("faster", "slower"), measured)
    if !isempty(changed)
        println(io, "\n**Faster or slower points**\n")
        println(io, "| RHS case | p | L | DOFs/field | Speedup | 95% interval | Verdict |")
        println(io, "|:--|--:|--:|--:|--:|:-:|:--|")
        for row in changed
            @printf(io, "| %s | %d | %d | %d | %.3fx | %.3f–%.3f | %s |\n", row.case, row.polydeg,
                    row.level, row.dofs, row.speedup, row.interval[1], row.interval[2], row.verdict)
        end
    end
    failed = filter(row -> haskey(row, :error), rows)
    if !isempty(failed)
        println(io, "\n**Failed points**\n")
        println(io, "| RHS case | p | L | Error |")
        println(io, "|:--|--:|--:|:--|")
        for row in failed
            println(io, "| ", row.case, " | ", row.polydeg, " | ", row.level, " | ",
                    replace(row.error, "|" => "\\|"), " |")
        end
    end
    write_footer(io, "point", baseline, target)
end

function comparison_data(baselines, targets, rows)
    b_env, t_env = baselines[1]["environment"], targets[1]["environment"]
    function point(row)
        data = Dict{String, Any}("case" => row.case, "polydeg" => row.polydeg, "level" => row.level,
                                 "sweeps" => row.sweeps)
        haskey(row, :error) && return merge(data, Dict("error" => row.error))
        return merge(data,
                     Dict("dofs_per_field" => row.dofs, "speedup" => row.speedup,
                          "interval" => collect(row.interval), "verdict" => row.verdict))
    end
    return Dict("schema" => 1, "noise_band" => NOISE_BAND,
                "baseline" => Dict(key => b_env[key] for key in ("commit", "measured_at", "build_url")),
                "target" => Dict("commit" => t_env["commit"], "gpu" => gpu_name(t_env),
                     "measured_at" => t_env["measured_at"], "build_url" => t_env["build_url"]),
                "points" => map(point, rows))
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
    scope = ENV["BENCHMARK_SCOPE"]
    scope in ("mini", "sweep") || error("Unknown benchmark scope: $scope")
    head, baseline = comparison_commits()
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
            # One Julia thread keeps the CPU-side parts reproducible.
            script = joinpath(@__DIR__, "runbenchmarks.jl")
            result(label, i) = joinpath(output, "$label-$i.json")
            for i in 1:PROCESSES, (label, project, commit) in (("baseline", env, baseline), ("target", @__DIR__, head))
                @info "Measurement $i of $PROCESSES" label commit
                run(`$(Base.julia_cmd()) --threads=1 --project=$project $script $(result(label, i)) $commit $scope`)
            end
            results = Dict(label => [read_result(result(label, i)) for i in 1:PROCESSES]
                           for label in ("baseline", "target"))
            baselines, targets = results["baseline"], results["target"]
            foreach(r -> validate_comparison(r, targets[1]), [baselines; targets[2:end]])
            rows = comparison_rows(scope, baselines, targets)
            if scope == "mini"
                text = sprint(io -> write_mini_report(io, baselines, targets, rows))
            else
                text = sprint(io -> write_sweep_report(io, baselines, targets, rows))
                write(joinpath(output, "comparison.json"), JSON.json(comparison_data(baselines, targets, rows)))
            end
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
