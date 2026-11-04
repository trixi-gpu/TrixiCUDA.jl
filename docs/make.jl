# TrixiCUDA.jl is not accessible through Julia's LOAD_PATH currently
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter
using TrixiCUDA

DocMeta.setdocmeta!(TrixiCUDA, :DocTestSetup, :(using TrixiCUDA); recursive = true)

makedocs(modules = [TrixiCUDA],
         sitename = "TrixiCUDA.jl",
         pages = [
             "Home" => "index.md",
             "API Reference" => "reference.md"
         ],
         format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                                  assets = ["assets/favicon.ico"]))

deploydocs(repo = "github.com/trixi-gpu/TrixiCUDA.jl",
           devbranch = "main",
           push_preview = true)
