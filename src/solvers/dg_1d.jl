# Everything related to a DG semidiscretization in 1D

# Functions end with `_kernel` are CUDA kernels that are going to be launed by the `@cuda` macro.

# Kernel for calculating flux along normal direction
function flux_kernel!(flux_arr, u, flux::Function, equations::AbstractEquations{1})
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2) && k <= size(u, 3))
        u_node = get_node_vars(u, equations, j, k)

        flux_node = flux(u_node, 1, equations)

        @inbounds begin
            for ii in axes(u, 1)
                flux_arr[ii, j, k] = flux_node[ii]
            end
        end
    end

    return nothing
end

# kernel for calculating weak form
function weak_form_kernel!(du, derivative_dhat, flux_arr, equations::AbstractEquations{1})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 3))
        @inbounds begin
            for ii in axes(du, 2)
                du[i, j, k] += derivative_dhat[j, ii] * flux_arr[i, ii, k]
            end
        end
    end

    return nothing
end

# Functions begin with `cuda_` are the functions that pack CUDA kernels together, 
# calling Tthem from the host (i.e., CPU) and running them on the device (i.e., GPU).

# Pack kernels for calculating volume integrals
function cuda_volume_integral!(du, u, mesh::TreeMesh{1}, nonconservative_terms, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::DGSEM)
    derivative_dhat = CuArray{Float32}(dg.basis.derivative_dhat)
    flux_arr = similar(u)

    size_arr = CuArray{Float32}(undef, size(u, 2), size(u, 3))

    flux_kernel = @cuda launch=false flux_kernel!(flux_arr, u, flux, equations)
    flux_kernel(flux_arr, u, flux, equations; configurator_2d(flux_kernel, size_arr)...)

    weak_form_kernel = @cuda launch=false weak_form_kernel!(du, derivative_dhat, flux_arr,
                                                            equations)
    weak_form_kernel(du, derivative_dhat, flux_arr, equations;
                     configurator_3d(weak_form_kernel, du)...,)

    return nothing
end
