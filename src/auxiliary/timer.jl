function pretty_time(t::Float64) # Float32?
    if t >= 1
        return @sprintf("%.2f s", t)
    elseif t >= 1e-3
        return @sprintf("%.2f ms", t*1e3)
    elseif t >= 1e-6
        return @sprintf("%.2f µs", t*1e6)
    else
        return @sprintf("%.2f ns", t*1e9)
    end
end

function pretty_bytes(bytes::Int64) # Int32?
    bytes, mb = Base.prettyprint_getunits(bytes, length(Base._mem_units), Int64(1024))
    unit = Base._mem_units[mb]
    if mb == 1 # less than KiB
        suffix = (bytes == 0 || bytes == 1) ? "" : "s" # 0 byte or 1 byte
        return @sprintf("%d byte%s", bytes, suffix)
    else # KiB, MiB, GiB, TiB, PiB
        return @sprintf("%.2f %s", bytes, unit)
    end
end

# Timer with device synchronization
macro timer(args...)
    arg_list = collect(args)

    # Determine if a description string was provided
    if length(arg_list) == 2 && arg_list[1] isa String
        desc = arg_list[1]
        expr = arg_list[2]
    elseif length(arg_list) == 1
        desc = 0
        expr = arg_list[1]
    else
        error("@timer usage error: use either @timer \"desc\" expr or @timer expr")
    end

    quote
        if $(esc(desc)) != 0
            @info $(esc(desc))
        end

        local val, cpu_time,
        cpu_alloc_size, cpu_gc_time, cpu_mem_stats,
        gpu_alloc_size, gpu_mem_time, gpu_mem_stats = CUDA.@timed $(esc(expr)) # synchronize in CUDA.@timed

        local cpu_alloc_count = Base.gc_alloc_count(cpu_mem_stats)
        local gpu_alloc_count = gpu_mem_stats.alloc_count

        println("Total time: ", pretty_time(cpu_time))

        # Loop over CPU and GPU data
        for (type, gctime, memtime, bytes, allocs) in (("CPU", cpu_gc_time, 0, cpu_alloc_size, cpu_alloc_count),
                                                       ("GPU", 0, gpu_mem_time, gpu_alloc_size, gpu_alloc_count))
            allocs, ma = Base.prettyprint_getunits(allocs, length(Base._cnt_units), Int64(1000))
            if ma == 1 # less than K
                suffix = (allocs == 0 || allocs == 1) ? "" : "s" # 0 alloc or 1 alloc
                @printf("%s allocation: %d%s alloc%s, ", type, allocs, Base._cnt_units[ma], suffix)
            else # K, M, G, T, P
                @printf("%s allocation: %.2f%s allocs, ", type, allocs, Base._cnt_units[ma])
            end
            print(pretty_bytes(bytes))
            if gctime > 0
                @printf(", %.2f%% gc time", 100 * gctime/cpu_time)
            end
            if memtime > 0
                @printf(", %.2f%% memmgmt time", 100 * memtime/cpu_time)
            end
            println()
        end
        println()
    end
end
