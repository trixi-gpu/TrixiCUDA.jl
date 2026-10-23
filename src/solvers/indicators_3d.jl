# Reuse Trixi's host-side indicator cache, including its thread-indexed buffers.
function create_cache(::Type{IndicatorHennemannGassner},
                      equations::AbstractEquations{3}, basis::LobattoLegendreBasisGPU)
    basis_cpu = LobattoLegendreBasis(real(basis), polydeg(basis))
    return create_cache(IndicatorHennemannGassner, equations, basis_cpu)
end
