# Grid of the `[benchmark sweep]` benchmark: each case is measured along a level sweep at a
# fixed polynomial degree and along a degree sweep at a fixed refinement level.
const LEVEL_SWEEP = (polydeg = 3, levels = Dict(1 => 1:15, 2 => 1:7, 3 => 1:5))
const DEGREE_SWEEP = (polydegs = 1:7, level = Dict(1 => 12, 2 => 6, 3 => 4))

# Cases that cannot run the full grid yet; remove each entry once its issue is fixed.
const SWEEP_EXCEPTIONS = Dict("advection_mortar_3d" => (; levels = 1:4),           # #262
                              "mhd_alfven_wave_3d" => (; polydegs = 1:6),          # #265
                              "mhd_alfven_wave_mortar_3d" => (; polydegs = 1:6),   # #265
                              "mhd_ec_3d" => (; polydegs = 1:6),                   # #265
                              "mhd_ec_shockcapturing_3d" => (; polydegs = 1:6))    # #266

case_ndims(name) = parse(Int, match(r"_(\d)d$", name)[1])

# Distinct (polydeg, level) points of a case, ordered by polydeg and then level. `sweeps`
# names the sweeps a point belongs to; the point at p = 3 and the degree sweep's level
# belongs to both.
function sweep_points(name)
    d = case_ndims(name)
    exception = get(SWEEP_EXCEPTIONS, name, (;))
    levels = get(exception, :levels, LEVEL_SWEEP.levels[d])
    polydegs = get(exception, :polydegs, DEGREE_SWEEP.polydegs)
    sweeps = Dict{Tuple{Int, Int}, Vector{String}}()
    foreach(L -> push!(get!(sweeps, (LEVEL_SWEEP.polydeg, L), String[]), "level"), levels)
    foreach(p -> push!(get!(sweeps, (p, DEGREE_SWEEP.level[d]), String[]), "degree"), polydegs)
    return [(; polydeg, level, sweeps = sweeps[(polydeg, level)])
            for (polydeg, level) in sort!(collect(keys(sweeps)))]
end
