using TOML, Printf

function write_report(io, baseline, target)
    baseline["schema"] == target["schema"] == 1 || error("Unknown result schema")
    b_env, t_env = baseline["environment"], target["environment"]
    # Comparisons must come from the same job/hardware and resolved dependency set.
    for field in ("julia", "gpu", "host", "runtime", "driver", "threads", "agent")
        b_env[field] == t_env[field] || error("Environment mismatch: $field")
    end
    baseline["parameters"] == target["parameters"] || error("Benchmark parameters differ")
    package_id = "5056661c-942a-472c-b3bc-01507f717a60"
    b_packages = filter(p -> first(p) != package_id, b_env["packages"])
    t_packages = filter(p -> first(p) != package_id, t_env["packages"])
    b_packages == t_packages || error("Resolved dependency versions differ")
    b, t = baseline["benchmarks"], target["benchmarks"]
    keys(b) == keys(t) || error("Benchmark case sets differ")
    println(io, "## GPU RHS benchmark comparison\n")
    println(io, "Baseline: `", b_env["commit"], "`  \nTarget: `", t_env["commit"], "`\n")
    println(io, "GPU: ", t_env["gpu"], "; Julia ", t_env["julia"],
            "; CUDA runtime ", t_env["runtime"], "; driver ", t_env["driver"], ".\n")
    println(io, "Median wall time with GPU synchronization; Float32, one Julia thread. ",
            "Compilation, input generation, transfers, and resetting du are excluded.\n")
    println(io, "| RHS case | DOFs/field | Baseline (ms) | Target (ms) | Time change | Speedup |")
    println(io, "|:--|--:|--:|--:|--:|--:|")
    for name in sort!(collect(keys(t)))
        base, head = b[name], t[name]
        for field in ("shape", "seed", "dofs_per_field")
            base[field] == head[field] || error("Input mismatch in $name: $field")
        end
        bt, tt = base["median_ns"], head["median_ns"]
        isfinite(bt) && isfinite(tt) && bt > 0 && tt > 0 || error("Invalid timings: $name")
        @printf(io, "| %s | %d | %.4f | %.4f | %+.2f%% | %.3fx |\n",
                name, head["dofs_per_field"], bt / 1e6, tt / 1e6,
                100 * (tt / bt - 1), bt / tt)
    end
    println(io, "\nNegative time change and speedup > 1 indicate improvement. ",
            "Informational only: no regression threshold is applied. ",
            "Small differences may be measurement noise.\n")
    println(io, "The target's benchmark definitions and seeded random inputs are used for both commits. ",
            "PR builds use the fetched target-branch tip; branch builds without PR metadata ",
            "use the default branch. Default-branch builds use the first parent. ",
            "Raw samples, environment details, and both manifests are attached as build artifacts.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("Usage: report.jl BASELINE.toml TARGET.toml REPORT.md")
    baseline, target = TOML.parsefile(ARGS[1]), TOML.parsefile(ARGS[2])
    # Validate the entire comparison before creating an output file.
    report = sprint(io -> write_report(io, baseline, target))
    write(ARGS[3], report)
    print(report)
end
