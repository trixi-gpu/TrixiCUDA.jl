# Everything about wrapping PTX instructions in Julia for better performance.

# Read-only (non-coherent) global loads via inline PTX. `ldg_nc(A, i)` returns 
# `A[i]`, but loads it with `ld.global.nc`, i.e. through the GPU's read-only data path.
@inline function _ldg_nc(addr::UInt64, ::Type{Float64})
    Base.llvmcall("""
        %v = call double asm "ld.global.nc.f64 \$0, [\$1];", "=d,l"(i64 %0)
        ret double %v
        """, Float64, Tuple{UInt64}, addr)
end

# The address is passed to the inline asm as a plain `UInt64` (via `reinterpret`)
# rather than as a pointer, because pointer syntax in `llvmcall` IR depends on the
# Julia/LLVM version: `ptr addrspace(1)` (opaque pointer) is rejected on Julia 1.10 
# and 1.11, where `llvmcall` still parses IR in typed mode. Switch to passing `ptr` 
# directly once the minimum supported Julia version accepts it.
@inline function ldg_nc(A::CUDA.CuDeviceArray{T}, i::Integer) where {T <: Float64}
    @boundscheck checkbounds(A, i)
    return _ldg_nc(reinterpret(UInt64, pointer(A, i)), T)
end
