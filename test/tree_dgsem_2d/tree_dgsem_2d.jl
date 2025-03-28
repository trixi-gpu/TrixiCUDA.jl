module DGSEMTree2D

using Trixi, TrixiCUDA
using Test

@testset "DGSEM Tree 2D" begin
    # Include all the existing DESEM tree mesh tests in 2D
    include("advection_basic.jl")
    include("advection_mortar.jl")

    include("euler_blob_mortar.jl")
    include("euler_ec.jl")
    include("euler_shock.jl")
    include("euler_source_terms_nonperiodic.jl")
    include("euler_source_terms.jl")

    include("eulermulti_ec.jl")
    include("eulermulti_es.jl")

    include("hypdiff_nonperiodic.jl")

    include("mhd_alfven_wave_mortar.jl")
    include("mhd_alfven_wave.jl")
    include("mhd_ec.jl")
    include("mhd_shock.jl")

    include("shallowwater_ec.jl")
    include("shallowwater_source_terms.jl")
    include("shawllowwater_source_terms_nonperiodic.jl")
end

end # module
