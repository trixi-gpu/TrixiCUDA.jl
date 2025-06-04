# This method gets called from OrdinaryDiffEq's `solve(...)`
function initialize!(cb::DiscreteCallback{Condition, Affect!}, u_ode::CuArray, t,
                     integrator) where {Condition, Affect! <: AnalysisCallback}
    semi = integrator.p
    du_ode = first(get_tmp_cache(integrator))

    # Get static size for `u_ode` and `du_ode`
    println("size of u_ode: ", size(u_ode))
    println("size of du_ode: ", size(du_ode))

    # Copy `u_ode` and `du_ode` from GPU back to CPU
    u_ode = Array(u_ode)
    du_ode = Array(du_ode)

    initialize!(cb, u_ode, du_ode, t, integrator, semi)
end

# function (analysis_callback::AnalysisCallback)(u_ode, du_ode, integrator, semi)
#     mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
#     @unpack dt, t = integrator
#     iter = integrator.stats.naccept

#     # Compute the percentage of the simulation that is done
#     t = integrator.t
#     t_initial = first(integrator.sol.prob.tspan)
#     t_final = last(integrator.sol.prob.tspan)
#     sim_time_percentage = (t - t_initial) / (t_final - t_initial) * 100

#     # Record performance measurements and compute performance index (PID)
#     runtime_since_last_analysis = 1.0e-9 * (time_ns() -
#                                             analysis_callback.start_time_last_analysis)
#     # PID is an MPI-aware measure of how much time per global degree of freedom (i.e., over all ranks
#     # and threads) and per `rhs!` evaluation is required. MPI-aware means that it essentially adds up 
#     # the time spent on each computing unit. Thus, in an ideally parallelized program, the PID should be constant
#     # independent of the number of MPI ranks or threads used, since, e.g., using 4x the number of ranks should
#     # divide the runtime on each rank by 4. See also the Trixi.jl docs ("Performance" section) for
#     # more information.
#     ncalls_rhs_since_last_analysis = (ncalls(semi.performance_counter)
#                                       -
#                                       analysis_callback.ncalls_rhs_last_analysis)
#     # This assumes that the same number of threads is used on each MPI rank.
#     performance_index = runtime_since_last_analysis * mpi_nranks() *
#                         Threads.nthreads() /
#                         (ndofsglobal(mesh, solver, cache)
#                          *
#                          ncalls_rhs_since_last_analysis)

#     # Compute the total runtime since the analysis callback has been initialized, in seconds
#     runtime_absolute = 1.0e-9 * (time_ns() - analysis_callback.start_time)

#     # Compute the relative runtime per thread as time spent in `rhs!` divided by the number of calls 
#     # to `rhs!` and the number of local degrees of freedom
#     # OBS! This computation must happen *after* the PID computation above, since `take!(...)`
#     #      will reset the number of calls to `rhs!`
#     runtime_relative = 1.0e-9 * take!(semi.performance_counter) * Threads.nthreads() /
#                        ndofs(semi)

#     # Compute the total time spent in garbage collection since the analysis callback has been
#     # initialized, in seconds
#     # Note: `Base.gc_time_ns()` is not part of the public Julia API but has been available at least
#     #        since Julia 1.6. Should this function be removed without replacement in a future Julia
#     #        release, just delete this analysis quantity from the callback.
#     # Source: https://github.com/JuliaLang/julia/blob/b540315cb4bd91e6f3a3e4ab8129a58556947628/base/timing.jl#L83-L84
#     gc_time_absolute = 1.0e-9 * (Base.gc_time_ns() - analysis_callback.start_gc_time)

#     # Compute the percentage of total time that was spent in garbage collection
#     gc_time_percentage = gc_time_absolute / runtime_absolute * 100

#     # Obtain the current memory usage of the Julia garbage collector, in MiB, i.e., the total size of
#     # objects in memory that have been allocated by the JIT compiler or the user code.
#     # Note: `Base.gc_live_bytes()` is not part of the public Julia API but has been available at least
#     #        since Julia 1.6. Should this function be removed without replacement in a future Julia
#     #        release, just delete this analysis quantity from the callback.
#     # Source: https://github.com/JuliaLang/julia/blob/b540315cb4bd91e6f3a3e4ab8129a58556947628/base/timing.jl#L86-L97
#     memory_use = Base.gc_live_bytes() / 2^20 # bytes -> MiB

#     @trixi_timeit timer() "analyze solution" begin
#         # General information
#         mpi_println()
#         mpi_println("─"^100)
#         mpi_println(" Simulation running '", get_name(equations), "' with ",
#                     summary(solver))
#         mpi_println("─"^100)
#         mpi_println(" #timesteps:     " * @sprintf("% 14d", iter) *
#                     "               " *
#                     " run time:       " * @sprintf("%10.8e s", runtime_absolute))
#         mpi_println(" Δt:             " * @sprintf("%10.8e", dt) *
#                     "               " *
#                     " └── GC time:    " *
#                     @sprintf("%10.8e s (%5.3f%%)", gc_time_absolute, gc_time_percentage))
#         mpi_println(rpad(" sim. time:      " *
#                          @sprintf("%10.8e (%5.3f%%)", t, sim_time_percentage), 46) *
#                     " time/DOF/rhs!:  " * @sprintf("%10.8e s", runtime_relative))
#         mpi_println("                 " * "              " *
#                     "               " *
#                     " PID:            " * @sprintf("%10.8e s", performance_index))
#         mpi_println(" #DOFs per field:" * @sprintf("% 14d", ndofsglobal(semi)) *
#                     "               " *
#                     " alloc'd memory: " * @sprintf("%14.3f MiB", memory_use))
#         mpi_println(" #elements:      " *
#                     @sprintf("% 14d", nelementsglobal(mesh, solver, cache)))

#         # Level information (only for AMR and/or non-uniform `TreeMesh`es)
#         print_level_information(integrator.opts.callback, mesh, solver, cache)
#         mpi_println()

#         # Open file for appending and store time step and time information
#         if mpi_isroot() && analysis_callback.save_analysis
#             io = open(joinpath(analysis_callback.output_directory,
#                                analysis_callback.analysis_filename), "a")
#             print(io, iter)
#             print(io, " ", t)
#             print(io, " ", dt)
#         else
#             io = devnull
#         end

#         # Calculate current time derivative (needed for semidiscrete entropy time derivative, residual, etc.)
#         # `integrator.f` is usually just a call to `rhs!`
#         # However, we want to allow users to modify the ODE RHS outside of Trixi.jl
#         # and allow us to pass a combined ODE RHS to OrdinaryDiffEq, e.g., for
#         # hyperbolic-parabolic systems.
#         @notimeit timer() integrator.f(du_ode, u_ode, semi, t)
#         u = wrap_array(u_ode, mesh, equations, solver, cache)
#         du = wrap_array(du_ode, mesh, equations, solver, cache)
#         # Compute l2_error, linf_error
#         analysis_callback(io, du, u, u_ode, t, semi)

#         mpi_println("─"^100)
#         mpi_println()

#         # Add line break and close analysis file if it was opened
#         if mpi_isroot() && analysis_callback.save_analysis
#             # This resolves a possible type instability introduced above, since `io`
#             # can either be an `IOStream` or `devnull`, but we know that it must be
#             # an `IOStream here`.
#             println(io::IOStream)
#             close(io::IOStream)
#         end
#     end

#     # avoid re-evaluating possible FSAL stages
#     u_modified!(integrator, false)

#     # Reset performance measurements
#     analysis_callback.start_time_last_analysis = time_ns()
#     analysis_callback.ncalls_rhs_last_analysis = ncalls(semi.performance_counter)

#     return nothing
# end
