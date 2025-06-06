# Iterate over tuples of analysis integrals in a type-stable way using "lispy tuple programming".
function analyze_integrals(analysis_integrals::NTuple{N, Any}, io,
                           du::CuArray, u::CuArray, t,
                           semi) where {N}

    # Extract the first analysis integral and process it; keep the remaining to be processed later
    quantity = first(analysis_integrals)
    remaining_quantities = Base.tail(analysis_integrals)

    # FIXME: This is a temporary workaround to avoid the scalar indexing issue.
    du = Array(du)
    u = Array(u)

    res = analyze(quantity, du, u, t, semi)[1]

    # Note that we default to single process (i.e., no MPI) in GPU computing environment
    @printf(" %-12s:", pretty_form_utf(quantity))
    @printf("  % 10.8e", res)
    print(io, " ", res)

    println()

    # Recursively call this method with the unprocessed integrals
    analyze_integrals(remaining_quantities, io, du, u, t, semi)
    return nothing
end
