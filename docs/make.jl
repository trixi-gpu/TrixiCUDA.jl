push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter
using TrixiCUDA

DocMeta.setdocmeta!(TrixiCUDA, :DocTestSetup, :(using TrixiCUDA); recursive = true)

# A benchmark sweep's docs job runs on a push event, which Documenter only deploys for the
# default branch. This config deploys that build to the pull request's preview instead.
struct SweepPreview <: Documenter.DeployConfig
    pr::String
end

function Documenter.deploy_folder(cfg::SweepPreview; repo_previews, branch_previews, kwargs...)
    return Documenter.DeployDecision(; all_ok = true, branch = branch_previews, repo = repo_previews,
                                     is_preview = true, subfolder = "previews/PR$(cfg.pr)")
end

Documenter.authentication_method(::SweepPreview) = Documenter.HTTPS

function Documenter.authenticated_repo_url(::SweepPreview)
    return "https://$(ENV["GITHUB_ACTOR"]):$(ENV["GITHUB_TOKEN"])@github.com/$(ENV["GITHUB_REPOSITORY"]).git"
end

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

# The heatmaps on the benchmark page load assets/sweep-data.js: the sweep comparison given in
# BENCHMARK_SWEEP_COMPARISON, or null when no sweep results are available.
comparison = get(ENV, "BENCHMARK_SWEEP_COMPARISON", "")
write(joinpath(@__DIR__, "build", "assets", "sweep-data.js"),
      "window.SWEEP_COMPARISON = ", isempty(comparison) ? "null" : read(comparison, String), ";\n")

# After a sweep, deploy to the PR preview named in BENCHMARK_SWEEP_PR; otherwise deploy as usual.
pr = get(ENV, "BENCHMARK_SWEEP_PR", "")
deploydocs(repo = "github.com/trixi-gpu/TrixiCUDA.jl",
           devbranch = "main",
           push_preview = true,
           deploy_config = isempty(pr) ? Documenter.auto_detect_deploy_system() : SweepPreview(pr))
