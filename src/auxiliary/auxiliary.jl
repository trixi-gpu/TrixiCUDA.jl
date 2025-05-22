include("configurators.jl")
include("stable.jl")

# Initialize the device 
function init_device()
    try
        # TODO: Consider multiple GPUs later
        device = CUDA.device()

        try
            # Get properties 
            global MULTIPROCESSOR_COUNT = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            global MAX_THREADS_PER_BLOCK = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            global MAX_SHARED_MEMORY_PER_BLOCK = CUDA.attribute(device, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
        catch e
            # Handle CUDA errors
            println("Error getting device properties: ", e.msg)
            println("Ensure that your GPU device properties can be retrieved successfully.")

            # Fall back to set zeros
            global MULTIPROCESSOR_COUNT = 0
            global MAX_THREADS_PER_BLOCK = 0
            global MAX_SHARED_MEMORY_PER_BLOCK = 0
            println("Device properties have been set to zero. CUDA operations 
            will fail because no valid properties are configured.")
        end

        try
            # Create CUDA streams
            global STREAM1 = CUDA.CuStream()
            global STREAM2 = CUDA.CuStream()
        catch e
            # Handle CUDA errors
            println("Error initializing CUDA streams: ", e.msg)
            println("Ensure there are enough GPU resources available for stream creation.")

            # Fall back to set default streams
            global STREAM1 = CUDA.stream()
            global STREAM2 = CUDA.stream()
            println("Streams have been set to the default CUDA stream. 
            This will impact performance due to no concurrency.")
        end

    catch e
        # Handle CUDA errors or not
        if e isa CUDA.CuError
            println("Error detecting device: ", e.msg)
            println("Ensure a CUDA compatible GPU is available and properly configured.")
        else
            println("An unexpected error occurred: ", e)
        end

        # Fall back to set nothing
        global MULTIPROCESSOR_COUNT = nothing
        global MAX_THREADS_PER_BLOCK = nothing
        global MAX_SHARED_MEMORY_PER_BLOCK = nothing
        global STREAM1 = nothing
        global STREAM2 = nothing
    end
end
