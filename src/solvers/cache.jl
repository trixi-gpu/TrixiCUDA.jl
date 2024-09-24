# Rewrite `create_cache` in Trixi.jl to add the specialized parts of the arrays 
# required to copy the arrays from CPU to GPU.

# See also `create_cache` in Trixi.jl
function create_cache_gpu(mesh::TreeMesh{1}, equations, dg::DGSEM, RealT, uEltype)
    # Get cells for which an element needs to be created (i.e. all leaf cells)

    ### ALL COPY TO GPU!!!
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

    boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

    cache = (; elements, interfaces, boundaries)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache = (; cache...,
             create_cache_gpu(mesh, equations, dg.volume_integral, dg, uEltype)...)

    return cache
end

# Note that `volume_integral::VolumeIntegralPureLGLFiniteVolume` is currently experimental
# in Trixi.jl and it is not implemented here.

function create_cache_gpu(mesh::TreeMesh{1}, equations,
                          volume_integral::VolumeIntegralWeakForm, dg::DGSEM, uEltype)
    NamedTuple()
end

function create_cache_gpu(mesh::TreeMesh{1}, equations,
                          volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM, uEltype)
    NamedTuple()
end

# function create_cache_gpu(mesh::TreeMesh{1}, equations,
#                       volume_integral::VolumeIntegralShockCapturingHG, dg::DGSEM, uEltype)
#     element_ids_dg = Int[]
#     element_ids_dgfv = Int[]

#     cache = create_cache(mesh, equations,
#                          VolumeIntegralFluxDifferencing(volume_integral.volume_flux_dg),
#                          dg, uEltype)

#     A2dp1_x = Array{uEltype, 2}
#     fstar1_L_threaded = A2dp1_x[A2dp1_x(undef, nvariables(equations), nnodes(dg) + 1)
#                                 for _ in 1:Threads.nthreads()]
#     fstar1_R_threaded = A2dp1_x[A2dp1_x(undef, nvariables(equations), nnodes(dg) + 1)
#                                 for _ in 1:Threads.nthreads()]

#     return (; cache..., element_ids_dg, element_ids_dgfv, fstar1_L_threaded,
#             fstar1_R_threaded)
# end
