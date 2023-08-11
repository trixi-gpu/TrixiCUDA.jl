# The header part for testing true nonconservative terms in 1D
equations = ShallowWaterEquations1D(gravity_constant=1.0f0, H0=3.0f0)

@changeprecision Float32 begin
    function initial_condition_well_balancedness(x, t, equations::ShallowWaterEquations1D)

        H = equations.H0
        v = 0.0

        b = (1.5 / exp(0.5 * ((x[1] - 1.0)^2)) + 0.75 / exp(0.5 * ((x[1] + 1.0)^2)))

        return prim2cons(SVector(H, v, b), equations)
    end
end

initial_condition = initial_condition_well_balancedness

boundary_condition = BoundaryConditionDirichlet(initial_condition)

volume_flux = (flux_wintermeyer_etal, flux_nonconservative_wintermeyer_etal)
solver = DGSEM(polydeg=4, surface_flux=(flux_hll, flux_nonconservative_fjordholm_etal),
    volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = 0.0f0
coordinates_max = Float32(sqrt(2.0))
mesh = TreeMesh(coordinates_min, coordinates_max,
    initial_refinement_level=3,
    n_cells_max=10_000,
    periodicity=false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
    boundary_conditions=boundary_condition)

@unpack mesh, equations, initial_condition, boundary_conditions, source_terms, solver, cache = semi

t = 0.0f0
tspan = (0.0f0, 100.0f0)

ode = semidiscretize(semi, tspan)
u_ode = copy(ode.u0)
du_ode = similar(u_ode)
u = wrap_array(u_ode, mesh, equations, solver, cache)
du = wrap_array(du_ode, mesh, equations, solver, cache)