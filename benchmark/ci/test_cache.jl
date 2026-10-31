using Test
include("results.jl")

function cache_fixture()
    env = Dict("commit" => "baseline-sha", "julia" => "1.10.12", "gpu" => "GPU",
               "gpu_uuid" => "UUID", "host" => "host", "runtime" => "13.4", "driver" => "13.3",
               "threads" => 1, "harness" => "hash", "measured_at" => "2026-10-30",
               "packages" => Dict("dependency" => Dict("version" => "1.0", "tree" => "tree")))
    cases = Dict("case" => Dict("minimum_ns" => 1.0, "samples_ns" => [1.0, 2.0], "samples" => 2,
                      "shape" => [1, 4, 8], "seed" => 123, "dofs_per_field" => 32))
    return Dict("schema" => RESULT_SCHEMA, "environment" => env,
                "parameters" => Dict("samples" => 100), "benchmarks" => cases, "reused" => false)
end

@testset "Exact baseline cache" begin
    mktempdir() do directory
        expected = cache_fixture()
        path = cache_path(directory, expected)
        @test endswith(path, ".json")
        @test restore_baseline(path, expected, ["case"]) === nothing
        write_result(path, expected)
        @test read_result(path) == expected
        restored = restore_baseline(path, expected, ["case"])
        @test restored["reused"]
        @test restored["environment"]["measured_at"] == "2026-10-30"
        @test restored["benchmarks"] == expected["benchmarks"]
        for field in ("commit", "julia", "gpu", "gpu_uuid", "host", "runtime", "driver", "threads", "harness")
            changed = deepcopy(expected)
            changed["environment"][field] = "different"
            @test cache_path(directory, changed) != path
            @test restore_baseline(path, changed, ["case"]) === nothing
        end
        changed = deepcopy(expected)
        changed["environment"]["packages"]["dependency"]["tree"] = "different-source"
        @test cache_path(directory, changed) != path
        @test restore_baseline(path, changed, ["case"]) === nothing
        changed = deepcopy(expected)
        changed["parameters"]["samples"] = 10
        @test cache_path(directory, changed) != path
        @test restore_baseline(path, changed, ["case"]) === nothing
        @test restore_baseline(path, expected, ["different-case"]) === nothing
        changed = deepcopy(expected)
        changed["environment"]["measured_at"] = "later"
        @test cache_path(directory, changed) == path
        old = deepcopy(expected)
        old["schema"] = RESULT_SCHEMA - 1
        write_result(path, old)
        @test_logs (:warn, r"Ignoring invalid baseline cache") restore_baseline(path, expected, ["case"])
        @test restore_baseline(path * ".missing", expected, ["case"]) === nothing
    end
end
