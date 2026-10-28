using Test, TOML
include("report.jl")

function report_fixture()
    environment = Dict("commit" => "baseline-sha", "julia" => "1.10.10",
                       "gpu" => "test GPU", "host" => "host", "runtime" => "13.0",
                       "driver" => "13.0", "threads" => 1, "agent" => "benchmark-agent",
                       "packages" => Dict("dependency" => Dict("name" => "Trixi", "version" => "0.17.16")))
    results = Dict(name => Dict("median_ns" => time, "shape" => [1, 4, 8],
                                "seed" => 123, "dofs_per_field" => 32)
                   for (name, time) in (("faster", 2e6), ("unchanged", 1e6), ("slower", 1e6)))
    return Dict("schema" => 1, "environment" => environment,
                "parameters" => Dict("samples" => 100), "benchmarks" => results)
end

@testset "Benchmark comparison report" begin
    baseline = report_fixture()
    target = deepcopy(baseline)
    target["environment"]["commit"] = "target-sha"
    target["benchmarks"]["faster"]["median_ns"] = 1e6
    target["benchmarks"]["slower"]["median_ns"] = 2e6
    # Test the serialized form used for real artifacts as well as the calculations.
    baseline = TOML.parse(sprint(io -> TOML.print(io, baseline)))
    target = TOML.parse(sprint(io -> TOML.print(io, target)))
    report = sprint(io -> write_report(io, baseline, target))
    @test occursin("| faster | 32 | 2.0000 | 1.0000 | -50.00% | 2.000x |", report)
    @test occursin("| unchanged | 32 | 1.0000 | 1.0000 | +0.00% | 1.000x |", report)
    @test occursin("| slower | 32 | 1.0000 | 2.0000 | +100.00% | 0.500x |", report)
    @test occursin("baseline-sha", report) && occursin("target-sha", report)
    for field in ("gpu", "host", "runtime", "julia", "driver", "threads", "agent", "packages")
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
        bad["benchmarks"]["faster"]["median_ns"] = value
        @test_throws Exception write_report(devnull, baseline, bad)
    end
    bad = deepcopy(target)
    delete!(bad["benchmarks"], "faster")
    @test_throws Exception write_report(devnull, baseline, bad)
    bad = deepcopy(target)
    bad["parameters"]["samples"] = 1
    @test_throws Exception write_report(devnull, baseline, bad)
end
