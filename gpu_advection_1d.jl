### Introduce CUDA.jl into the implementation of discretization of 1D advection
### equation 
### Note that this is also a raw implementation and the code is not yet optimized

using Trixi, LinearAlgebra, OrdinaryDiffEq, CUDA

# Get CPU scalar data
coordinates_min = -1.0
coordinates_max = 1.0
n_elements      = 16    # Suppose an extreme large number
polydeg         = 3     # Suppose an extreme large number
dx              = (coordinates_max - coordinates_min) / n_elements

# Create basis with Lagrange polynomials
basis = LobattoLegendreBasis(polydeg)

# --------------------------------------------------------------------- GPU Acceleration----------     # Number type checking is uncompleted
# Get CPU array data and copy to GPU
nodes = CuArray(basis.nodes)
D = CuArray(basis.derivative_matrix)
M = CuArray(diagm(basis.weights))
B = CuArray(diagm([-1; zeros(polydeg - 1); 1]))

# Create column vector x_{l} on GPU                             # May need a custom kernel?
x_l = CuArray(collect(0:n_elements-1)) .* dx + CUDA.fill(dx / 2 - 1, n_elements)

# Get matrix x from column vectors x_{l} and ξ on GPU           # May need a custom kernel?
x = CUDA.fill(1.0, polydeg + 1) * transpose(x_l) + nodes * transpose(CUDA.fill(1.0, n_elements)) .* (dx / 2)

# Write a CUDA kernel for u(x, 0) (i.e. u0(x))
# Choose to design grid size as (blocks_x, blocks_y, 1) and block size as (16, 16, 1)
# (Note that it may not be the perfect design and it is necessary to experiment with 
# different combinations of grid and block sizes to determine the most efficient configuration)
function initial_condition_sine_wave_gpu!(u0, x)
	# Get thread index i and j 
	index_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	index_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

	# Create corresponding stride for index i and j
	stride_i = gridDim().x * blockDim().x
	stride_j = gridDim().y * blockDim().y

	# Assign each thread to calculate corresponding u0
	for i ∈ index_i:stride_i:n_elements
		for j ∈ index_j:stride_j:(polydeg+1)
			@inbounds u0[i, j] = 1.0 + 0.5 * sin(pi * x[i, j])
		end
	end

	return nothing
end

# Design the number of blocks in x and y directions
blocks_x = ceil(Int, n_elements / 16)
blocks_y = ceil(Int, (polydeg + 1) / 16)

# Apply kernel function to matrix x to get u0
u0 = similar(x)
@cuda threads = (16, 16) blocks = (blocks_x, blocks_y) initial_condition_sine_wave_gpu!(u0, x)

# Write a CUDA kernel for numerical flux
# Choose to design grid size as (blocks_num, 1, 1) and block size as (16, 1, 1)
# (Note that it may not be the perfect design and it is necessary to experiment with 
# different combinations of grid and block sizes to determine the most efficient configuration)
function apply_surface_flux!(γ, u1, u2)
	# Get thread index 
	index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

	# Get loop stride
	stride = gridDim().x * blockDim().x

	# Choose to use Lax-Friedrichs flux and set equations
	surface_flux = flux_lax_friedrichs
	equations = LinearScalarAdvectionEquation1D(1.0)

	# Assign each thread to calculate corresponding flux
	for i ∈ index:stride:n_elements
		@inbounds γ[i] = surface_flux(u1[i], u2[i], 1, equations)
	end

	return nothing
end

# Apply the above CUDA kernel within rhs!() function            
function rhs!(du, u, x, t)
	# Reset CPU data and copy data to GPU
	u = CuArray(u)
	du = CuArray{eltype(du)}(undef, (polydeg + 1, n_elements))

	# Caculate u_{0} and u_{N} on GPU                           # May need a custom kernel?
	u_0 = transpose([1.0; CUDA.fill(0.0, polydeg)]) * u * CuArray(Matrix{Float64}(I, n_elements, n_elements)[:, [2:n_elements; 1]])
	u_N = transpose([CUDA.fill(0.0, polydeg); 1.0]) * u

	# Design the number of blocks in x direction
	blocks_num = ceil(Int, n_elements / 16)

	# Apply kernel function to vectors u_{0} and u_{N}
	γ = similar(du)
	@cuda threads = 16 blocks = blocks_num apply_surface_flux!(γ, u_0, u_N)

	# Get flux matrix λ from γ and γ' on GPU                 # May need a custom kernel?
	λ = [γ * CuArray(Matrix{Float64}(I, n_elements, n_elements)[:, [n_elements; 1:n_elements-1]]); CUDA.zeros(polydeg - 1, n_elements); γ]

	# Reset matrix du on GPU                                   # May need a custom kernel?
	du = (-(M \ B) * λ + (M \ transpose(D)) * M * u) .* (2 / dx)

	# Copy GPU data back to CPU
	du = Array(du)

	return nothing
end

# Copy GPU data back to CPU
u0 = Array(u0)
# ------------------------------------------------------------------------------------------------

# Create and solve ODE problem
tspan = (0.0, 2.0)
ode = ODEProblem(rhs!, u0, tspan)
sol = solve(ode, RDPK3SpFSAL49(), abstol = 1.0e-6, reltol = 1.0e-6, save_everystep = false)
