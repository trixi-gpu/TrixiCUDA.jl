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
             create_cache_gpu(mesh, equations, dg.volume_integral, dg, uEltype, cache)...)

    return cache
end

# Note that `volume_integral::VolumeIntegralPureLGLFiniteVolume` is currently experimental
# in Trixi.jl and it is not implemented here.

# Create cache for 1D tree mesh
function create_cache_gpu(mesh::TreeMesh{1}, equations,
                          volume_integral::VolumeIntegralWeakForm, dg::DGSEM,
                          uEltype, cache)
    NamedTuple()
end

function create_cache_gpu(mesh::TreeMesh{1}, equations,
                          volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM,
                          uEltype, cache)
    NamedTuple()
end

function create_cache_gpu(mesh::TreeMesh{1}, equations,
                          volume_integral::VolumeIntegralShockCapturingHG, dg::DGSEM,
                          uEltype, cache)
    fstar1_L = CUDA.zeros(Float64, nvariables(equations), nnodes(dg) + 1, nelements(cache.elements))
    fstar1_R = CUDA.zeros(Float64, nvariables(equations), nnodes(dg) + 1, nelements(cache.elements))

    cache = create_cache_gpu(mesh, equations,
                             VolumeIntegralFluxDifferencing(volume_integral.volume_flux_dg),
                             dg, uEltype, cache)

    # Remove `element_ids_dg` and `element_ids_dgfv` here
    return (; cache..., fstar1_L, fstar1_R)
end
