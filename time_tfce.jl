include("synthetic.jl")
include("tfce.jl")
include("tfce2.jl")

using BenchmarkTools

Z_3d = generate_3d_data((100, 100, 100))

z_min = minimum(Z_3d)
z_max = maximum(Z_3d)
range_val = z_max - z_min
dh_equivalent = range_val / 50

# Run and time tfce_3d for positive clusters only.
result = tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent)
@time result = tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent)

result = tfce_3d_unionfind(Z_3d; E=0.5, H=2.0)
GC.gc()
@time result = tfce_3d_unionfind(Z_3d; E=0.5, H=2.0)
nothing