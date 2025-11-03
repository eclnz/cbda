include("synthetic.jl")
include("tfce.jl")
include("tfce2.jl")

using Profile
using BenchmarkTools
Z_3d = generate_3d_data((100, 100, 100))

z_min = minimum(Z_3d)
z_max = maximum(Z_3d)
using ProfileSVG

range_val = z_max - z_min
dh_equivalent = range_val / 50
tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent)
Profile.clear()
@profile tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent)
ProfileSVG.save("tfce_profile.svg")
open("dfont", "w") do f
    Profile.print(IOContext(f, :color => false))
end

nx, ny, nz = size(Z_3d)
n_voxels = nx * ny * nz
t_stat_flat = reshape(Z_3d, n_voxels)
tfce_flat = zeros(Float64, n_voxels)
t_min = convert(Float64, minimum(Z_3d))
t_max = convert(Float64, maximum(t_stat_flat))
dh_T = convert(Float64, dh_equivalent)
h = dh_T
E = 0.5
H = 2.0
height_part = (h - t_min)^H

visited = zeros(Bool, n_voxels)
frontier = Vector{Int}(undef, n_voxels)
cluster = Vector{Int}(undef, n_voxels)
frontier_len = 0
cidx = div(n_voxels, 2)

using InteractiveUtils

println("Running benchmarks...")
@time tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent, skip=10)
@time tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent)

@btime tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent, skip=10)
@btime tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent)
@btime tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent)

result2 = tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent, skip=10)
result1 = tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent)
are_close = isapprox(result2, result1; rtol=1e-5, atol=1e-8)
diff = result2 .- result1
maxdiff = maximum(abs.(diff))
meandiff = mean(abs.(diff))

# Percentage differences, handling near-zero values
abs1 = abs.(result1)
# Use a mask to avoid division by ~zero; ignore values below a small threshold in denominator
eps_mask = abs1 .> 1e-12
pct_diff = similar(result1)
pct_diff[eps_mask] .= abs.(diff[eps_mask]) ./ abs1[eps_mask] .* 100
pct_diff[.!eps_mask] .= 0.0
maxpctdiff = maximum(pct_diff)
meanpctdiff = mean(pct_diff)

println("Are the results close? ", are_close)
println("Maximum absolute difference: ", maxdiff)
println("Mean absolute difference: ", meandiff)
println("Maximum percentage difference: $(maxpctdiff)%")
println("Mean percentage difference: $(meanpctdiff)%")


println