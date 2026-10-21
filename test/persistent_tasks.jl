using Pkg

# Aqua 0.8.18 reconstructs dependencies from package Project.toml files, losing
# registry corrections such as CUDA_Runtime_jll's CUDA_Compiler_jll dependency.
# Use the resolved test manifest while checking precompilation and task shutdown.
function precompile_finishes_from_manifest(package::Module; tmax = 30)
    mktempdir() do directory
        package_id = Base.PkgId(package)
        wrapper_name = "TrixiCUDAPrecompileCheck"
        write(joinpath(directory, "Project.toml"), """
              name = "$wrapper_name"
              uuid = "2b90b810-3847-44f4-89af-b1f7d499a628"
              version = "0.1.0"

              [deps]
              $(package_id.name) = "$(package_id.uuid)"
              """)
        manifest = Pkg.Types.EnvCache().manifest_file
        cp(manifest, joinpath(directory, "Manifest.toml"))
        preferences = joinpath(dirname(Base.active_project()), "LocalPreferences.toml")
        if isfile(preferences)
            cp(preferences, joinpath(directory, "LocalPreferences.toml"))
        end

        ready = joinpath(directory, "loaded")
        mkpath(joinpath(directory, "src"))
        write(joinpath(directory, "src", "$wrapper_name.jl"), """
              module $wrapper_name
              using $(package_id.name)
              touch($(repr(ready)))
              end
              """)
        log_path = joinpath(directory, "precompile.log")
        expression = "using Pkg; Pkg.precompile(; already_instantiated = true)"
        command = `$(Base.julia_cmd()) --startup-file=no --project=$directory -e $expression`
        open(log_path, "w") do log
            process = run(pipeline(command; stdout = log, stderr = log); wait = false)
            try
                timedwait(() -> isfile(ready) || !process_running(process), Inf; pollint = 0.5)
                if !isfile(ready)
                    wait(process)
                    error("Precompilation failed before loading $(package_id.name):\n" *
                          read(log_path, String))
                end
                finished = timedwait(() -> !process_running(process), tmax; pollint = 0.1) == :ok
                if finished
                    wait(process)
                    success(process) || error("Precompilation failed:\n" * read(log_path, String))
                end
                return finished
            finally
                if process_running(process)
                    kill(process, Base.SIGKILL)
                    wait(process)
                end
            end
        end
    end
end
