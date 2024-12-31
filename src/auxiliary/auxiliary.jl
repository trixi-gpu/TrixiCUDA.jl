include("configurators.jl")
include("stable.jl")

# Initialize the device properties
function init_device()
    # Consider single GPU for now
    # TODO: Consider multiple GPUs later
    device = CUDA.device()

    # Get the device properties
    global MULTIPROCESSOR_COUNT = CUDA.attribute(device,
                                                 CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    global MAX_THREADS_PER_BLOCK = CUDA.attribute(device,
                                                  CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
end
