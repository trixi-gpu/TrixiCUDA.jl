# Create some macros to simplify the testing process.

# Load the required packages, macros, and structs
using Trixi, TrixiCUDA
using Test: @test, @testset

# Macro to test the type Float64 or Float32 ?

# Macro to test the approximate equality of arrays from GPU and CPU, while 
# also handling the cases related to NaNs.
macro test_approx(expr)
    # Parse the expression and check that it is of the form array1 ≈ array2
    if expr.head != :call || expr.args[1] != :≈
        error("Usage: @test_approx array1 ≈ array2")
    end

    local gpu = esc(expr.args[2])
    local cpu = esc(expr.args[3])

    quote
        # Convert to arrays to avoid using CUDA.@allowscalar 
        # to access the elements of some arrays
        local gpu_arr = Array($gpu)
        local cpu_arr = Array($cpu)

        # Check if the arrays have NaN
        local has_nan_gpu = any(isnan, gpu_arr)
        local has_nan_cpu = any(isnan, cpu_arr)

        if has_nan_gpu && has_nan_cpu # both have NaN
            # Condition 1: Check if NaNs are at the same position
            local cond1 = isnan.(gpu_arr) == isnan.(cpu_arr)

            # Replace NaNs with 0.0
            local _gpu_arr = replace(gpu_arr, NaN => 0.0)
            local _cpu_arr = replace(cpu_arr, NaN => 0.0)

            # Condition 2: Check if the arrays are approximately equal
            local cond2 = _gpu_arr ≈ _cpu_arr

            @test cond1 && cond2
        elseif !has_nan_gpu && !has_nan_cpu # neither has NaN

            # Direct comparison
            @test gpu_arr ≈ cpu_arr
        else # one has NaN and the other does not

            # Typically, the array from CPU has NaN and the array from 
            # GPU does not have NaN, since the NaN values are replaced 
            # with zeros in the GPU kernels to avoid control flow divergence 
            # when dealing with NaNs. 

            # Condition 1's truth table:
            # -------------------------------
            # Entry-CPU | Entry-GPU | Result
            # -------------------------------
            #  NaN      |   zero    |   1
            #  NaN      |  non-zero |   0
            #  non-NaN  |   zero    |   1
            #  non-NaN  |  non-zero |   1
            # -------------------------------
            local cond1 = all(.!(isnan.(cpu_arr) .&& (gpu_arr .!= 0.0)))

            # Replace NaNs with 0.0
            local _cpu_arr = replace(cpu_arr, NaN => 0.0)

            # Condition 2: Check if the arrays are approximately equal
            local cond2 = gpu_arr ≈ _cpu_arr

            @test cond1 && cond2
        end
    end
end
