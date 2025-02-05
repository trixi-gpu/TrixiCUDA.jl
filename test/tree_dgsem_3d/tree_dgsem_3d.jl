module DGSEMTree3D

using Trixi, TrixiCUDA
using Test

@testset "DGSEM Tree 3D" begin
    # Include all the existing DESEM tree mesh tests in 3D
    include("advection_basic.jl")
    include("advection_mortar.jl")

    include("euler_convergence.jl")
    include("euler_ec.jl")
    include("euler_mortar.jl")
    include("euler_shock.jl")
    include("euler_source_terms.jl")

    include("hypdiff_nonperiodic.jl")

    include("mhd_alfven_wave_mortar.jl")
    include("mhd_alfven_wave.jl")
    include("mhd_ec.jl")
    include("mhd_shock.jl")
end

end # module
