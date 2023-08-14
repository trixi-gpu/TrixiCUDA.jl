using CUDA
import Adapt

mutable struct Interpolate{A}
    xs::A
    ys::A
end

function (itp::Interpolate)(x)
    i = searchsortedfirst(itp.xs, x)
    i = clamp(i, firstindex(itp.ys), lastindex(itp.ys))
    @inbounds itp.ys[i]
end

xs_cpu = [1.0, 2.0, 3.0]
ys_cpu = [10.0, 20.0, 30.0]
itp_cpu = Interpolate(xs_cpu, ys_cpu)
pts_cpu = [1.1, 2.3]
result_cpu = itp_cpu.(pts_cpu)

function Adapt.adapt_structure(to, itp::Interpolate)
    xs = Adapt.adapt_structure(to, itp.xs)
    ys = Adapt.adapt_structure(to, itp.ys)
    Interpolate(xs, ys)
end

itp = Interpolate(CuArray(xs_cpu), CuArray(ys_cpu))
pts = CuArray(pts_cpu)

result = itp.(pts)