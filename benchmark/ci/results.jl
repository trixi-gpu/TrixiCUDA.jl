using JSON, SHA, TOML

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
