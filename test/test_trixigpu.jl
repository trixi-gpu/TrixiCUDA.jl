# Create some macros to simplify the testing process.
using Test: @test
using CUDA: @allowscalar

# Macro to test the approximate equality of arrays from GPU and CPU, while 
# also handling the cases related to NaNs.
macro test_approx(expr)
    # Parse the expression and check that it is of the form a ≈ b
    if expr.head != :call || expr.args[1] != :≈
        error("Usage: @test_approx a ≈ b")
    end

    local gpu_arr = expr.args[2]
    local cpu_arr = expr.args[3]

    quote
        # Check if the arrays have NaN
        local has_nan_gpu = any(isnan, $gpu_arr)
        local has_nan_cpu = any(isnan, $cpu_arr)

        if has_nan_gpu && has_nan_cpu # both have NaN
            # Condition 1: Check if NaNs are at the same position
            local cond1 = isnan.($gpu_arr) == isnan.($cpu_arr)

            # Replace NaNs with 0.0
            local _gpu_arr = replace($gpu_arr, NaN => 0.0)
            local _cpu_arr = replace($cpu_arr, NaN => 0.0)

            # Condition 2: Check if the arrays are approximately equal
            local cond2 = @allowscalar _gpu_arr ≈ _cpu_arr

            @test cond1 && cond2
        elseif !has_nan_gpu && !has_nan_cpu # neither has NaN

            # Direct comparison
            @test @allowscalar $gpu_arr ≈ $cpu_arr
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
            local cond1 = all(.!(isnan.($cpu_arr) .&& ($gpu_arr .!= 0.0)))

            # Replace NaNs with 0.0
            local _cpu_arr = replace($cpu_arr, NaN => 0.0)

            # Condition 2: Check if the arrays are approximately equal
            local cond2 = @allowscalar $gpu_arr ≈ _cpu_arr

            @test cond1 && cond2
        end
    end
end

# todo: check gpu array indexing source - and remove all @allowscalar
