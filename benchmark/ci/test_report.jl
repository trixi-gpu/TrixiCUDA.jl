using Test
include("report.jl")

function report_fixture()
    environment = Dict("commit" => "baseline-sha", "julia" => "1.10.10",
                       "gpu" => "test GPU", "host" => "host", "runtime" => "13.0",
                       "driver" => "13.0", "threads" => 1, "agent" => "benchmark-agent",
                       "gpu_uuid" => "device-uuid", "harness" => "harness-hash",
                       "measured_at" => "2026-10-30T00:00:00", "build_url" => "local",
                       "packages" => Dict("dependency" => Dict("name" => "Trixi", "version" => "0.17.16")))
    results = Dict(name => Dict("minimum_ns" => time, "samples_ns" => [time, 2time],
                        "samples" => 2, "shape" => [1, 4, 8],
                        "seed" => 123, "dofs_per_field" => 32)
                   for (name, time) in (("faster", 2e6), ("unchanged", 1e6), ("slower", 1e6)))
    return Dict("schema" => RESULT_SCHEMA, "environment" => environment, "reused" => false,
                "parameters" => Dict("samples" => 100), "benchmarks" => results)
end

@testset "Benchmark comparison report" begin
    baseline = report_fixture()
    target = deepcopy(baseline)
    target["environment"]["commit"] = "target-sha"
    for (name, time) in (("faster", 1e6), ("slower", 2e6))
        target["benchmarks"][name]["minimum_ns"] = time
        target["benchmarks"][name]["samples_ns"] = [time, 2time]
    end
    # Test the serialized form used for real artifacts as well as the calculations.
    baseline, target = mktempdir() do directory
        files = joinpath.(directory, ("baseline.json", "target.json"))
        write_result(files[1], baseline)
        write_result(files[2], target)
        @test startswith(read(files[1], String), "{")
        @test startswith(read(files[2], String), "{")
        read_result.(files)
    end
    report = sprint(io -> write_report(io, baseline, target))
    @test occursin("| faster | 32 | 2.0000 | 1.0000 | -50.00% | 2.000x |", report)
    @test occursin("| unchanged | 32 | 1.0000 | 1.0000 | +0.00% | 1.000x |", report)
    @test occursin("| slower | 32 | 1.0000 | 2.0000 | +100.00% | 0.500x |", report)
    @test occursin("baseline-sha", report) && occursin("target-sha", report)
    @test occursin("Minimum wall time", report)
    @test occursin("geometric mean speedup: **1.000x**", report)
    @test !occursin("Median", report)
    @test occursin("baseline.json and target.json", report)
    cached = deepcopy(baseline)
    cached["reused"] = true
    @test occursin("Baseline measurement: reused", sprint(io -> write_report(io, cached, target)))
    for field in ("gpu", "gpu_uuid", "host", "runtime", "julia", "driver", "threads", "packages", "harness")
        bad = deepcopy(target)
        bad["environment"][field] = "mismatch"
        @test_throws Exception write_report(devnull, baseline, bad)
    end
    for field in ("shape", "seed", "dofs_per_field")
        bad = deepcopy(target)
        bad["benchmarks"]["faster"][field] = nothing
        @test_throws Exception write_report(devnull, baseline, bad)
    end
    for value in (0.0, -1.0, Inf, NaN)
        bad = deepcopy(target)
        bad["benchmarks"]["faster"]["minimum_ns"] = value
        @test_throws Exception write_report(devnull, baseline, bad)
    end
    bad = deepcopy(target)
    delete!(bad["benchmarks"], "faster")
    @test_throws Exception write_report(devnull, baseline, bad)
    bad = deepcopy(target)
    bad["parameters"]["samples"] = 1
    @test_throws Exception write_report(devnull, baseline, bad)
end
