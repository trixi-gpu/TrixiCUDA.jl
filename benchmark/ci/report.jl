using Printf
include("results.jl")

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

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error("Usage: report.jl BASELINE.json TARGET.json REPORT.md")
    baseline, target = read_result(ARGS[1]), read_result(ARGS[2])
    # Validate the entire comparison before creating an output file.
    report = sprint(io -> write_report(io, baseline, target))
    write(ARGS[3], report)
    print(report)
end
