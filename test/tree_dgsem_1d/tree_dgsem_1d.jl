module TreeDGSEM1D

include("advection_basic.jl")
include("advection_extended.jl")
include("burgers_basic.jl")
include("burgers_rarefraction.jl")
include("burgers_shock.jl")
include("euler_blast_wave.jl")
include("euler_ec.jl")
include("euler_shock.jl")
include("euler_source_terms_nonperiodic.jl")
include("euler_source_terms.jl")
include("eulermulti_ec.jl")
include("eulermulti_es.jl")
include("eulerquasi_ec.jl")
include("eulerquasi_source_terms.jl")
include("hypdiff_harmonic_nonperiodic.jl")
include("hypdiff_nonperiodic.jl")
include("mhd_alfven_wave.jl")
include("mhd_ec.jl")
include("shallowwater_shock.jl")

end # module 
