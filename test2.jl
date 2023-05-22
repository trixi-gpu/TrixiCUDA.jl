using Trixi, LinearAlgebra, OrdinaryDiffEq, CUDA, Test

coordinates_min = -1.0
coordinates_max = 1.0
n_elements      = 16
polydeg         = 3
dx              = (coordinates_max - coordinates_min) / n_elements

basis = LobattoLegendreBasis(polydeg)

nodes = CuArray{Float32}(basis.nodes)
D = CuArray(convert(Array{Float32}, basis.derivative_matrix))
M = CuArray(convert(Array{Float32}, diagm(basis.weights)))
B = CuArray(convert(Array{Float32}, diagm([-1; zeros(polydeg - 1); 1])))

x_l = CuArray{Float32}(collect(0:n_elements-1)) .* Float32(dx) + CuArray{Float32}(fill(dx / 2 - 1, n_elements))

x = CuArray{Float32}(fill(1.0, polydeg + 1)) * transpose(x_l) + nodes * transpose(CuArray{Float32}(fill(1, n_elements))) .* (Float32(dx) / 2)

u0 = map(x -> Float32(0.5) * sin(pi * x) + 1, x)

function apply_surface_flux!(γ, u1, u2)
	index = threadIdx().x
	stride = blockDim().x

	surface_flux = flux_lax_friedrichs
	equations = LinearScalarAdvectionEquation1D(1.0)

	for i ∈ index:stride:length(u1)
		@inbounds γ[i] = surface_flux(u1[i], u2[i], 1, equations)
	end

	return nothing
end

u = CUDA.fill(1.0f0, (4, 16))
u_0 = CuArray(transpose([1.0f0; CUDA.fill(0.0f0, polydeg)]) * u * CuArray(Matrix{Float32}(I, n_elements, n_elements)[:, [2:n_elements; 1]]))
u_N = CuArray(transpose([CUDA.fill(0.0f0, polydeg); 1.0f0]) * u)
γ = similar(u_N)

@cuda threads = 8 apply_surface_flux!(γ, u_0, u_N)









#= function rhs!(du, u, x, t)
	u = CuArray(convert(Array{Float32}, u))
	du = CuArray{Float32}(undef, (polydeg + 1, n_elements))

	u_0 = transpose([1.0f0; CUDA.fill(0.0f0, polydeg)]) * u * CuArray(Matrix{Float32}(I, n_elements, n_elements)[:, [2:n_elements; 1]])
	u_N = transpose([CUDA.fill(0.0f0, polydeg); 1.0f0]) * u

	γ = similar(u_0)
	@cuda threads = 8 apply_surface_flux!(γ, u_0, u_N)

	λ = [γ * CuArray(Matrix{Float32}(I, n_elements, n_elements)[:, [n_elements; 1:n_elements-1]]); CUDA.zeros(polydeg - 1, n_elements); γ]

	du = (-(M \ B) * λ + (M \ transpose(D)) * M * u) .* (2 / dx)

	du = Array(du)

	return nothing
end

u0 = Array(u0)

tspan = (0.0f0, 2.0f0)
ode = ODEProblem(rhs!, u0, tspan)
sol = solve(ode, RDPK3SpFSAL49(), abstol = 1.0e-6, reltol = 1.0e-6, save_everystep = false)
 =#
