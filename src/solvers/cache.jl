# Rewrite `create_cache` in Trixi.jl to add the specialized parts of the arrays 
# required to copy the arrays from CPU to GPU.

# Note that `volume_integral::VolumeIntegralPureLGLFiniteVolume` is currently 
# experimental in Trixi.jl and it is not implemented here.

# Create cache for general tree mesh
function create_cache_gpu(mesh, equations,
                          volume_integral::VolumeIntegralWeakForm, dg::DG,
                          uEltype, cache)
    NamedTuple()
end

# Create cache specialized for 1D tree mesh
function create_cache_gpu(mesh::TreeMesh{1}, equations, dg::DG, RealT, uEltype)
    # Get cells for which an element needs to be created (i.e., all leaf cells)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

    boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

    # Copy cache components from CPU to GPU
    elements = copy_elements(elements)
    interfaces = copy_interfaces(interfaces)
    boundaries = copy_boundaries(boundaries)

    # GPU cache
    cache_gpu = (; elements, interfaces, boundaries)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache = (; cache_gpu...,
             create_cache_gpu(mesh, equations, dg.volume_integral, dg, uEltype, cache_gpu)...)

    return cache
end

function create_cache_gpu(mesh::TreeMesh{1}, equations,
                          volume_integral::VolumeIntegralFluxDifferencing, dg::DG,
                          uEltype, cache)
    NamedTuple()
end

function create_cache_gpu(mesh::TreeMesh{1}, equations,
                          volume_integral::VolumeIntegralShockCapturingHG, dg::DG,
                          uEltype, cache)
    fstar1_L = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg) + 1, nelements(cache.elements))
    fstar1_R = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg) + 1, nelements(cache.elements))

    cache = create_cache_gpu(mesh, equations,
                             VolumeIntegralFluxDifferencing(volume_integral.volume_flux_dg),
                             dg, uEltype, cache)

    return (; cache..., fstar1_L, fstar1_R)
end

# Create cache specialized for 2D tree mesh
function create_cache_gpu(mesh::TreeMesh{2}, equations,
                          dg::DG, RealT, uEltype)
    # Get cells for which an element needs to be created (i.e., all leaf cells)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

    boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

    mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    # Copy cache components from CPU to GPU
    elements = copy_elements(elements)
    interfaces = copy_interfaces(interfaces)
    boundaries = copy_boundaries(boundaries)
    mortars = copy_mortars(mortars)

    # GPU cache
    cache_gpu = (; elements, interfaces, boundaries, mortars)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache_gpu = (; cache_gpu...,
                 create_cache_gpu(mesh, equations, dg.volume_integral, dg, uEltype, cache_gpu)...)
    cache = (; cache_gpu..., create_cache_gpu(mesh, equations, dg.mortar, uEltype, cache_gpu)...)

    return cache
end

function create_cache_gpu(mesh::TreeMesh{2}, equations,
                          volume_integral::VolumeIntegralFluxDifferencing, dg::DG,
                          uEltype, cache)
    NamedTuple()
end

function create_cache_gpu(mesh::TreeMesh{2}, equations,
                          volume_integral::VolumeIntegralShockCapturingHG, dg::DG,
                          uEltype, cache)
    fstar1_L = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg) + 1, nnodes(dg),
                          nelements(cache.elements))
    fstar1_R = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg) + 1, nnodes(dg),
                          nelements(cache.elements))
    fstar2_L = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg), nnodes(dg) + 1,
                          nelements(cache.elements))
    fstar2_R = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg), nnodes(dg) + 1,
                          nelements(cache.elements))

    cache = create_cache_gpu(mesh, equations,
                             VolumeIntegralFluxDifferencing(volume_integral.volume_flux_dg),
                             dg, uEltype, cache)

    return (; cache..., fstar1_L, fstar1_R, fstar2_L, fstar2_R)
end

function create_cache_gpu(mesh::TreeMesh{2}, equations,
                          mortar_l2::LobattoLegendreMortarL2, uEltype, cache)
    fstar_primary_upper = CUDA.zeros(uEltype, nvariables(equations), nnodes(mortar_l2),
                                     nmortars(cache.mortars))
    fstar_primary_lower = CUDA.zeros(uEltype, nvariables(equations), nnodes(mortar_l2),
                                     nmortars(cache.mortars))
    fstar_secondary_upper = CUDA.zeros(uEltype, nvariables(equations), nnodes(mortar_l2),
                                       nmortars(cache.mortars))
    fstar_secondary_lower = CUDA.zeros(uEltype, nvariables(equations), nnodes(mortar_l2),
                                       nmortars(cache.mortars))

    (; fstar_primary_upper, fstar_primary_lower, fstar_secondary_upper, fstar_secondary_lower)
end

# Create cache specialized for 3D tree mesh
function create_cache_gpu(mesh::TreeMesh{3}, equations,
                          dg::DG, RealT, uEltype)
    # Get cells for which an element needs to be created (i.e., all leaf cells)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

    boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

    mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    # Copy cache components from CPU to GPU
    elements = copy_elements(elements)
    interfaces = copy_interfaces(interfaces)
    boundaries = copy_boundaries(boundaries)
    mortars = copy_mortars(mortars)

    # GPU cache
    cache_gpu = (; elements, interfaces, boundaries, mortars)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache_gpu = (; cache_gpu...,
                 create_cache_gpu(mesh, equations, dg.volume_integral, dg, uEltype, cache_gpu)...)
    cache = (; cache_gpu..., create_cache_gpu(mesh, equations, dg.mortar, uEltype, cache_gpu)...)

    return cache
end

function create_cache_gpu(mesh::TreeMesh{3}, equations,
                          volume_integral::VolumeIntegralFluxDifferencing, dg::DG,
                          uEltype, cache)
    NamedTuple()
end

function create_cache_gpu(mesh::TreeMesh{3}, equations,
                          volume_integral::VolumeIntegralShockCapturingHG, dg::DG,
                          uEltype, cache)
    fstar1_L = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg) + 1, nnodes(dg),
                          nnodes(dg), nelements(cache.elements))
    fstar1_R = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg) + 1, nnodes(dg),
                          nnodes(dg), nelements(cache.elements))
    fstar2_L = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg), nnodes(dg) + 1,
                          nnodes(dg), nelements(cache.elements))
    fstar2_R = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg), nnodes(dg) + 1,
                          nnodes(dg), nelements(cache.elements))
    fstar3_L = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg), nnodes(dg),
                          nnodes(dg) + 1, nelements(cache.elements))
    fstar3_R = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg), nnodes(dg),
                          nnodes(dg) + 1, nelements(cache.elements))

    cache = create_cache_gpu(mesh, equations,
                             VolumeIntegralFluxDifferencing(volume_integral.volume_flux_dg),
                             dg, uEltype, cache)

    return (; cache..., fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R)
end

function create_cache_gpu(mesh::TreeMesh{3}, equations,
                          mortar_l2::LobattoLegendreMortarL2, uEltype, cache)
    fstar_primary_upper_left = CUDA.zeros(uEltype, nvariables(equations), nnodes(mortar_l2),
                                          nnodes(mortar_l2), nmortars(cache.mortars))
    fstar_primary_upper_right = CUDA.zeros(uEltype, nvariables(equations), nnodes(mortar_l2),
                                           nnodes(mortar_l2), nmortars(cache.mortars))
    fstar_primary_lower_left = CUDA.zeros(uEltype, nvariables(equations), nnodes(mortar_l2),
                                          nnodes(mortar_l2), nmortars(cache.mortars))
    fstar_primary_lower_right = CUDA.zeros(uEltype, nvariables(equations), nnodes(mortar_l2),
                                           nnodes(mortar_l2), nmortars(cache.mortars))
    fstar_secondary_upper_left = CUDA.zeros(uEltype, nvariables(equations), nnodes(mortar_l2),
                                            nnodes(mortar_l2), nmortars(cache.mortars))
    fstar_secondary_upper_right = CUDA.zeros(uEltype, nvariables(equations), nnodes(mortar_l2),
                                             nnodes(mortar_l2), nmortars(cache.mortars))
    fstar_secondary_lower_left = CUDA.zeros(uEltype, nvariables(equations), nnodes(mortar_l2),
                                            nnodes(mortar_l2), nmortars(cache.mortars))
    fstar_secondary_lower_right = CUDA.zeros(uEltype, nvariables(equations), nnodes(mortar_l2),
                                             nnodes(mortar_l2), nmortars(cache.mortars))

    # Temporary arrays can also be created here
    (; fstar_primary_upper_left, fstar_primary_upper_right, fstar_primary_lower_left,
     fstar_primary_lower_right, fstar_secondary_upper_left, fstar_secondary_upper_right,
     fstar_secondary_lower_left, fstar_secondary_lower_right)
end
