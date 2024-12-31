include("configurators.jl")
include("stable.jl")

# Initialize the device properties
function init_device()
    try
        # Consider single GPU for now
        # TODO: Consider multiple GPUs later
        device = CUDA.device()

        # Get the device properties
        global MULTIPROCESSOR_COUNT = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        global MAX_THREADS_PER_BLOCK = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    catch e
        # Handle the errors
        if e isa CUDA.CuError
            println("Error initializing device: ", e.msg)
            println("Ensure a CUDA-enabled GPU is available and properly configured.")
        else
            println("An unexpected error occurred: ", e)
        end

        # Fall back to set default values
        global MULTIPROCESSOR_COUNT = 0
        global MAX_THREADS_PER_BLOCK = 0
    end
end
