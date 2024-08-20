#!/usr/bin/env julia

using Pkg
Pkg.activate(; temp = true, io = devnull)
Pkg.add(PackageSpec(name = "JuliaFormatter", version = "1.0.56"); preserve = PRESERVE_ALL,
        io = devnull)

using JuliaFormatter: format

function main()
    # Get the file path from the command line, default to the current directory
    if isempty(ARGS)
        paths = String["."]
    else
        paths = ARGS
    end

    # Try to apply formatting to the specified file or directory
    try
        format(paths)
    catch e
        println("Error formatting! Usage: format.jl PATH [PATH...]")
    end
end

main()
