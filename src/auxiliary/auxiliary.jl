include("configurators.jl")
include("stable.jl")

# Initialize the device 
function init_device()
    # Get the device properties
    try
        # TODO: Consider multiple GPUs later
        device = CUDA.device()

        global MULTIPROCESSOR_COUNT = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        global MAX_THREADS_PER_BLOCK = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        global MAX_SHARED_MEMORY_PER_BLOCK = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
        # Catch the errors
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
        global MAX_SHARED_MEMORY_PER_BLOCK = 0
    end

    # Initialize the CUDA streams
    try
        global STREAM1 = CUDA.CuStream()
        global STREAM2 = CUDA.CuStream()
        # Catch the errors
    catch e
        println("Error initializing CUDA streams: ", e)
        println("Ensure there are enough GPU resources for stream creation.")

        # Fallback to default stream
        global STREAM1 = CUDA.stream()
        global STREAM2 = CUDA.stream()

        println("Using the default stream as a fallback.")
    end
end
