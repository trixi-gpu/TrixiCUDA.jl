# Rewrite `create_cache` in Trixi.jl to add the specialized parts of the arrays 
# required to copy the arrays from CPU to GPU.

# Note that `volume_integral::VolumeIntegralPureLGLFiniteVolume` is currently 
# experimental in Trixi.jl and it is not implemented here.

# Create cache for general tree mesh
function create_cache_gpu(mesh, equations,
                          volume_integral::VolumeIntegralWeakForm, dg::DGSEM,
                          uEltype, cache)
    NamedTuple()
end

# Create cache specialized for 1D tree mesh
function create_cache_gpu(mesh::TreeMesh{1}, equations, dg::DGSEM, RealT, uEltype)
    # Get cells for which an element needs to be created (i.e. all leaf cells)
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

# Create cache specialized for 2D tree mesh
function create_cache_gpu(mesh::TreeMesh{2}, equations,
                          dg::DGSEM, RealT, uEltype)
    # Get cells for which an element needs to be created (i.e. all leaf cells)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

    boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

    mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    cache = (; elements, interfaces, boundaries, mortars)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache = (; cache...,
             create_cache_gpu(mesh, equations, dg.volume_integral, dg, uEltype, cache)...)
    cache = (; cache..., create_cache_gpu(mesh, equations, dg.mortar, uEltype, cache)...)

    return cache
end

function create_cache_gpu(mesh::TreeMesh{2}, equations,
                          volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM,
                          uEltype, cache)
    NamedTuple()
end

function create_cache_gpu(mesh::TreeMesh{2}, equations,
                          volume_integral::VolumeIntegralShockCapturingHG, dg::DGSEM,
                          uEltype, cache)
    fstar1_L = CUDA.zeros(Float64, nvariables(equations), nnodes(dg) + 1, nnodes(dg),
                          nelements(cache.elements))
    fstar1_R = CUDA.zeros(Float64, nvariables(equations), nnodes(dg) + 1, nnodes(dg),
                          nelements(cache.elements))
    fstar2_L = CUDA.zeros(Float64, nvariables(equations), nnodes(dg), nnodes(dg) + 1,
                          nelements(cache.elements))
    fstar2_R = CUDA.zeros(Float64, nvariables(equations), nnodes(dg), nnodes(dg) + 1,
                          nelements(cache.elements))

    cache = create_cache_gpu(mesh, equations,
                             VolumeIntegralFluxDifferencing(volume_integral.volume_flux_dg),
                             dg, uEltype, cache)

    return (; cache..., fstar1_L, fstar1_R, fstar2_L, fstar2_R)
end

function create_cache_gpu(mesh::TreeMesh{2}, equations,
                          mortar_l2::LobattoLegendreMortarL2, uEltype, cache)
    fstar_upper = CUDA.zeros(Float64, nvariables(equations), nnodes(mortar_l2),
                             nmortars(cache.mortars))
    fstar_lower = CUDA.zeros(Float64, nvariables(equations), nnodes(mortar_l2),
                             nmortars(cache.mortars))

    (; fstar_upper, fstar_lower)
end

# Create cache specialized for 3D tree mesh
function create_cache_gpu(mesh::TreeMesh{3}, equations,
                          dg::DGSEM, RealT, uEltype)
    # Get cells for which an element needs to be created (i.e. all leaf cells)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

    boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

    mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    cache = (; elements, interfaces, boundaries, mortars)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache = (; cache...,
             create_cache_gpu(mesh, equations, dg.volume_integral, dg, uEltype, cache)...)
    cache = (; cache..., create_cache_gpu(mesh, equations, dg.mortar, uEltype, cache)...)

    return cache
end

function create_cache_gpu(mesh::TreeMesh{3}, equations,
                          volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM,
                          uEltype, cache)
    NamedTuple()
end

function create_cache_gpu(mesh::TreeMesh{3}, equations,
                          volume_integral::VolumeIntegralShockCapturingHG, dg::DGSEM,
                          uEltype, cache)
    fstar1_L = CUDA.zeros(Float64, nvariables(equations), nnodes(dg) + 1, nnodes(dg),
                          nnodes(dg), nelements(cache.elements))
    fstar1_R = CUDA.zeros(Float64, nvariables(equations), nnodes(dg) + 1, nnodes(dg),
                          nnodes(dg), nelements(cache.elements))
    fstar2_L = CUDA.zeros(Float64, nvariables(equations), nnodes(dg), nnodes(dg) + 1,
                          nnodes(dg), nelements(cache.elements))
    fstar2_R = CUDA.zeros(Float64, nvariables(equations), nnodes(dg), nnodes(dg) + 1,
                          nnodes(dg), nelements(cache.elements))
    fstar3_L = CUDA.zeros(Float64, nvariables(equations), nnodes(dg), nnodes(dg),
                          nnodes(dg) + 1, nelements(cache.elements))
    fstar3_R = CUDA.zeros(Float64, nvariables(equations), nnodes(dg), nnodes(dg),
                          nnodes(dg) + 1, nelements(cache.elements))

    cache = create_cache_gpu(mesh, equations,
                             VolumeIntegralFluxDifferencing(volume_integral.volume_flux_dg),
                             dg, uEltype, cache)

    return (; cache..., fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R)
end

function create_cache_gpu(mesh::TreeMesh{3}, equations,
                          mortar_l2::LobattoLegendreMortarL2, uEltype, cache)
    fstar_upper_left = CUDA.zeros(Float64, nvariables(equations), nnodes(mortar_l2),
                                  nnodes(mortar_l2), nmortars(cache.mortars))
    fstar_upper_right = CUDA.zeros(Float64, nvariables(equations), nnodes(mortar_l2),
                                   nnodes(mortar_l2), nmortars(cache.mortars))
    fstar_lower_left = CUDA.zeros(Float64, nvariables(equations), nnodes(mortar_l2),
                                  nnodes(mortar_l2), nmortars(cache.mortars))
    fstar_lower_right = CUDA.zeros(Float64, nvariables(equations), nnodes(mortar_l2),
                                   nnodes(mortar_l2), nmortars(cache.mortars))

    # Temporary arrays can also be created here
    (; fstar_upper_left, fstar_upper_right, fstar_lower_left, fstar_lower_right)
end
