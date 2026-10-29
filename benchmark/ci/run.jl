using TrixiCUDA, CUDA
using BenchmarkTools, Statistics, TOML, Pkg
include("cases.jl")

function benchmark_rhs(case; seconds, samples, seed)
    semi = make_problem(case, DGSEMGPU, SemidiscretizationHyperbolicGPU)
    (; mesh, equations, boundary_conditions, source_terms, solver, cache_gpu, cache_cpu) = semi
    u_host = random_input(case, equations, solver, cache_gpu, seed)
    u = CuArray(u_host)
    du = CUDA.zeros(Float32, size(u))
    t = 0.0f0

    # Compile and populate caches before measuring. Fail on invalid random states
    # instead of publishing plausible timings for NaN-producing computations.
    for _ in 1:2
        fill!(du, 0.0f0)
        CUDA.@sync TrixiCUDA.rhs_gpu!(du, u, t, mesh, equations, boundary_conditions,
                                      source_terms, solver, cache_gpu, cache_cpu)
    end
    all(isfinite, Array(du)) || error("Non-finite RHS in $(case.name)")

    # Synchronize the reset outside the timed expression. evals=1 guarantees each
    # measurement starts with du=0, including RHS implementations that accumulate.
    # The block keeps BenchmarkTools options outside CUDA.@sync.
    trial = @benchmark begin
        CUDA.@sync TrixiCUDA.rhs_gpu!($du, $u, $t, $mesh, $equations,
                                      $boundary_conditions, $source_terms,
                                      $solver, $cache_gpu, $cache_cpu)
    end setup=(CUDA.@sync fill!($du, 0.0f0)) evals=1 seconds=seconds samples=samples
    return Dict("median_ns" => median(trial).time,
                "minimum_ns" => minimum(trial).time,
                "samples_ns" => trial.times,
                "samples" => length(trial),
                "dofs_per_field" => length(u) ÷ Trixi.nvariables(equations),
                "shape" => collect(size(u)), "seed" => seed)
end

function main()
    length(ARGS) == 2 || error("Usage: run.jl OUTPUT.toml COMMIT_SHA")
    CUDA.functional() || error("A functional CUDA GPU is required for RHS benchmarks")
    seconds = parse(Float64, get(ENV, "TRIXICUDA_BENCHMARK_SECONDS", "2"))
    samples = parse(Int, get(ENV, "TRIXICUDA_BENCHMARK_SAMPLES", "100"))
    seconds > 0 && samples > 0 || error("Benchmark duration and samples must be positive")
    filter = get(ENV, "TRIXICUDA_BENCHMARK_FILTER", "")
    cases = [c for c in rhs_cases() if occursin(filter, c.name)]
    isempty(cases) && error("No benchmark cases match '$filter'")
    # This setting is also used by the shock-capturing path, which copies to CPU.
    Threads.nthreads() == 1 || error("Run with --threads=1 for reproducible comparisons")
    packages = Dict(string(id) => Dict("name" => info.name, "version" => string(info.version))
                    for (id, info) in Pkg.dependencies() if info.version !== nothing)
    environment = Dict("commit" => ARGS[2], "julia" => string(VERSION),
                       "gpu" => CUDA.name(CUDA.device()), "host" => gethostname(),
                       "runtime" => string(CUDA.runtime_version()),
                       "driver" => string(CUDA.driver_version()),
                       "threads" => Threads.nthreads(), "packages" => packages,
                       "cuda_versioninfo" => sprint(CUDA.versioninfo),
                       "agent" => get(ENV, "BUILDKITE_AGENT_NAME", "local"))
    results = Dict{String, Any}()
    for case in cases
        @info "Benchmarking $(case.name)"
        results[case.name] = benchmark_rhs(case; seconds, samples, seed = 123)
        # Avoid retaining all case allocations on small benchmark GPUs.
        GC.gc(true)
        CUDA.reclaim()
    end
    output = Dict("schema" => 1, "environment" => environment,
                  "parameters" => Dict("seconds" => seconds, "samples" => samples,
                                       "precision" => "Float32", "evals" => 1,
                                       "filter" => filter),
                  "benchmarks" => results)
    open(ARGS[1], "w") do io
        TOML.print(io, output; sorted = true)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
