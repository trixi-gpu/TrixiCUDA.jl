# TrixiCUDA.jl is not accessible through Julia's LOAD_PATH currently
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter
using TrixiCUDA

DocMeta.setdocmeta!(TrixiCUDA, :DocTestSetup, :(using TrixiCUDA); recursive = true)

# The sweep heatmaps read the comparison of a sweep run, if one is given.
comparison = get(ENV, "BENCHMARK_SWEEP_COMPARISON", "")
write(joinpath(@__DIR__, "src", "assets", "benchmarks", "sweep-data.js"),
      "window.SWEEP_COMPARISON = ", isempty(comparison) ? "null" : read(comparison, String), ";\n")

makedocs(modules = [TrixiCUDA],
         sitename = "TrixiCUDA.jl",
         pages = [
             "Home" => "index.md",
             "Benchmark Results" => "benchmark.md"
         ],
         format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                                  assets = ["assets/favicon.ico",
                                      "assets/heatmaps.css",
                                      "assets/heatmaps.js"]))

deploydocs(repo = "github.com/trixi-gpu/TrixiCUDA.jl",
           devbranch = "main",
           push_preview = true)
