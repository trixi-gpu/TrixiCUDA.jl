using CUDA, Test, BenchmarkTools

firsts = [1, 2, 3, 4]
lasts = [5, 6, 7, 8]

lasts_firsts = CuArray{Int32}(firsts[1]:lasts[4])
indices_arr = CuArray{Int32}([firsts[1], firsts[2], firsts[3], firsts[4]])