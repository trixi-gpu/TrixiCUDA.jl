function SemidiscretizationHyperbolic_gpu(mesh, equations, initial_condition, solver;
                                          source_terms = nothing,
                                          boundary_conditions = boundary_condition_periodic,
                                          # `RealT` is used as real type for node locations etc.
                                          # while `uEltype` is used as element type of solutions etc.
                                          RealT = real(solver), uEltype = RealT,
                                          initial_cache = NamedTuple())
    @assert ndims(mesh) == ndims(equations)

    # cache = (; create_cache(mesh, equations, solver, RealT, uEltype)...,
    #          initial_cache...)
    # _boundary_conditions = digest_boundary_conditions(boundary_conditions, mesh, solver,
    #                                                   cache)

    # check_periodicity_mesh_boundary_conditions(mesh, _boundary_conditions)

    # SemidiscretizationHyperbolic{typeof(mesh), typeof(equations),
    #                              typeof(initial_condition),
    #                              typeof(_boundary_conditions), typeof(source_terms),
    #                              typeof(solver), typeof(cache)}(mesh, equations,
    #                                                             initial_condition,
    #                                                             _boundary_conditions,
    #                                                             source_terms, solver,
    #                                                             cache)
end
