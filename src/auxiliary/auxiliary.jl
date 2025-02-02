include("configurators.jl")
include("stable.jl")

# Initialize the device 
function init_device()
    try
        # TODO: Consider multiple GPUs later
        device = CUDA.device()

        # Get properties 
        global MULTIPROCESSOR_COUNT = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        global MAX_THREADS_PER_BLOCK = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        global MAX_SHARED_MEMORY_PER_BLOCK = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)

        # Create CUDA streams
        global STREAM1 = CUDA.CuStream()
        global STREAM2 = CUDA.CuStream()
    catch e
        # Handle the errors
        if e isa CUDA.CuError
            println("Error initializing device: ", e.msg)
            println("Ensure a CUDA compatible GPU is available and properly configured.")
        else
            println("An unexpected error occurred: ", e)
        end

        # Fall back to set default values
        global MULTIPROCESSOR_COUNT = 0
        global MAX_THREADS_PER_BLOCK = 0
        global MAX_SHARED_MEMORY_PER_BLOCK = 0
        global STREAM1 = nothing
        global STREAM2 = nothing
    end
end
