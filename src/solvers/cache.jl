# Everything about cache that stores both host data and device data.
 
# Note that `volume_integral::VolumeIntegralPureLGLFiniteVolume` is currently experimental 
# in Trixi.jl and it is not implemented here.

# TODO: Need to check the performance of `HG` separately for 1D, 2D and 3D about design.

function create_cache_gpu(mesh, equations,
                          volume_integral::VolumeIntegralWeakForm, dg::DGSEM,
                          uEltype, cache)
    NamedTuple()
end

# Create cache specialized for 1D `TreeMesh`.
function create_cache_gpu(mesh::TreeMesh{1}, equations, dg::DGSEM, RealT, uEltype)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    basis_cpu = LobattoLegendreBasis(real(dg), polydeg(dg))
    elements = init_elements(leaf_cell_ids, mesh, equations, basis_cpu, RealT, uEltype)
    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)
    boundaries = init_boundaries(leaf_cell_ids, mesh, elements, basis_cpu)

    # Copy cache components from CPU to GPU.
    # TODO: This copy could possible by overlapping with some GPU kernels execution.
    # And this optimizatinon could largely improve the performance if they are not 
    # used early in RHS.
    elements = copy_elements(elements)
    interfaces = copy_interfaces(interfaces)
    boundaries = copy_boundaries(boundaries)

    # Container cache
    cache = (; elements, interfaces, boundaries)
    # Add volume integral cache
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
                          volume_integral::VolumeIntegralShockCapturingHGType, dg::DGSEM,
                          uEltype, cache)
    fstar1_L = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg) + 1, nelements(cache.elements))
    fstar1_R = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg) + 1, nelements(cache.elements))

    # Device buffers for the indicator values and their smoothing.
    alpha = CuArray{real(dg)}(undef, nelements(cache.elements))
    alpha_tmp = similar(alpha)

    cache = create_cache_gpu(mesh, equations,
                             volume_integral.volume_integral_blend_high_order,
                             dg, uEltype, cache)

    return (; cache..., fstar1_L, fstar1_R, alpha, alpha_tmp)
end

# Create cache specialized for 2D `TreeMesh`.
function create_cache_gpu(mesh::TreeMesh{2}, equations, dg::DGSEM, RealT, uEltype)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    basis_cpu = LobattoLegendreBasis(real(dg), polydeg(dg))
    elements = init_elements(leaf_cell_ids, mesh, equations, basis_cpu, RealT, uEltype)
    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)
    boundaries = init_boundaries(leaf_cell_ids, mesh, elements, basis_cpu)
    mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    # Copy cache components from CPU to GPU.
    # TODO: This copy could possible by overlapping with some GPU kernels execution.
    # And this optimizatinon could largely improve the performance if they are not 
    # used early in RHS.
    elements = copy_elements(elements)
    interfaces = copy_interfaces(interfaces)
    boundaries = copy_boundaries(boundaries)
    mortars = copy_mortars(mortars)

    # Container cache
    cache = (; elements, interfaces, boundaries, mortars)
    # Add volume integral cache
    cache = (; cache...,
             create_cache_gpu(mesh, equations, dg.volume_integral, dg, uEltype, cache)...)
    # Add mortar cache
    cache = (; cache..., create_cache_gpu(mesh, equations, dg.mortar, uEltype, cache)...)

    return cache
end

function create_cache_gpu(mesh::TreeMesh{2}, equations,
                          volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM,
                          uEltype, cache)
    NamedTuple()
end

function create_cache_gpu(mesh::TreeMesh{2}, equations,
                          volume_integral::VolumeIntegralShockCapturingHGType, dg::DGSEM,
                          uEltype, cache)
    fstar1_L = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg) + 1, nnodes(dg),
                          nelements(cache.elements))
    fstar1_R = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg) + 1, nnodes(dg),
                          nelements(cache.elements))
    fstar2_L = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg), nnodes(dg) + 1,
                          nelements(cache.elements))
    fstar2_R = CUDA.zeros(uEltype, nvariables(equations), nnodes(dg), nnodes(dg) + 1,
                          nelements(cache.elements))

    # Device buffers for the indicator values and their smoothing.
    alpha = CuArray{real(dg)}(undef, nelements(cache.elements))
    alpha_tmp = similar(alpha)

    cache = create_cache_gpu(mesh, equations,
                             volume_integral.volume_integral_blend_high_order,
                             dg, uEltype, cache)

    return (; cache..., fstar1_L, fstar1_R, fstar2_L, fstar2_R, alpha, alpha_tmp)
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

# Create cache specialized for 3D `TreeMesh`.
function create_cache_gpu(mesh::TreeMesh{3}, equations, dg::DGSEM, RealT, uEltype)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    basis_cpu = LobattoLegendreBasis(real(dg), polydeg(dg))
    elements = init_elements(leaf_cell_ids, mesh, equations, basis_cpu, RealT, uEltype)
    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)
    boundaries = init_boundaries(leaf_cell_ids, mesh, elements, basis_cpu)
    mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    # Copy cache components from CPU to GPU
    # TODO: This copy could possible by overlapping with some GPU kernels execution.
    # And this optimizatinon could largely improve the performance if they are not 
    # used early in RHS.
    elements = copy_elements(elements)
    interfaces = copy_interfaces(interfaces)
    boundaries = copy_boundaries(boundaries)
    mortars = copy_mortars(mortars)

    # Container cache
    cache = (; elements, interfaces, boundaries, mortars)
    # Add volume integral cache
    cache = (; cache...,
             create_cache_gpu(mesh, equations, dg.volume_integral, dg, uEltype, cache)...)
    # Add mortar cache
    cache = (; cache..., create_cache_gpu(mesh, equations, dg.mortar, uEltype, cache)...)

    return cache
end

function create_cache_gpu(mesh::TreeMesh{3}, equations,
                          volume_integral::VolumeIntegralFluxDifferencing, dg::DGSEM,
                          uEltype, cache)
    NamedTuple()
end

function create_cache_gpu(mesh::TreeMesh{3}, equations,
                          volume_integral::VolumeIntegralShockCapturingHGType, dg::DGSEM,
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

    # Device buffers for the indicator values and their smoothing.
    alpha = CuArray{real(dg)}(undef, nelements(cache.elements))
    alpha_tmp = similar(alpha)

    cache = create_cache_gpu(mesh, equations,
                             volume_integral.volume_integral_blend_high_order,
                             dg, uEltype, cache)

    return (; cache..., fstar1_L, fstar1_R, fstar2_L, fstar2_R, fstar3_L, fstar3_R, alpha, alpha_tmp)
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

    # Temporary arrays can also be created here.
    (; fstar_primary_upper_left, fstar_primary_upper_right, fstar_primary_lower_left,
     fstar_primary_lower_right, fstar_secondary_upper_left, fstar_secondary_upper_right,
     fstar_secondary_lower_left, fstar_secondary_lower_right)
end
