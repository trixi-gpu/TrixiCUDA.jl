using Pkg

# Two independent environments, with identical dependency versions. If the
# baseline cannot use that resolution, fail rather than compare different stacks.
length(ARGS) == 4 || error("Usage: prepare.jl BENCHMARK_PROJECT TARGET BASELINE OUTPUT_DIR")
project, target, baseline, output = abspath.(ARGS)
withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
    for (name, source) in (("target", target), ("baseline", baseline))
        env = joinpath(output, name)
        mkpath(env)
        cp(joinpath(project, "Project.toml"), joinpath(env, "Project.toml"); force = true)
        if name == "target"
            rm(joinpath(env, "Manifest.toml"); force = true)
        end
        if name == "baseline"
            cp(joinpath(output, "target", "Manifest.toml"), joinpath(env, "Manifest.toml"); force = true)
        end
        Pkg.activate(env)
        Pkg.develop(PackageSpec(path = source); preserve = Pkg.PRESERVE_ALL)
        Pkg.instantiate()
        Pkg.status(; mode = Pkg.PKGMODE_MANIFEST)
    end
end
