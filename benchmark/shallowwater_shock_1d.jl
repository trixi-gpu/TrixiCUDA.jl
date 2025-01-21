using Trixi, TrixiCUDA
using CUDA
using BenchmarkTools

# Set the precision
RealT = Float32

# Set up the problem
