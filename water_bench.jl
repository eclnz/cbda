using Random
using Statistics
using ImageFiltering
using ImageMorphology
using Plots
using LinearAlgebra
using LoopVectorization
using BenchmarkTools

include("synthetic.jl")
include("water.jl")

Z_smooth = generate_2d_data(150)

levels, masks = flood_fill_levels(Z_smooth)

tfce_map = tfce_like(Z_smooth)
tfce_map_inplace = zeros(size(Z_smooth))
flooded_buf = Matrix{Bool}(undef, size(Z_smooth))
labeled_buf = Matrix{Int}(undef, size(Z_smooth))
tfce_like!(tfce_map_inplace, Z_smooth, flooded_buf, labeled_buf)

@assert tfce_map ≈ tfce_map_inplace "TFCE maps don't match"

println("2D Benchmarks:")
@time tfce_map = tfce_like(Z_smooth)
fill!(tfce_map_inplace, 0.0)
GC.gc()
@time tfce_like!(tfce_map_inplace, Z_smooth, flooded_buf, labeled_buf)

Z_3d_data = generate_3d_data((80, 80, 80))
nx_3d, ny_3d, nz_3d = size(Z_3d_data)

tfce_3d_result = tfce_like(Z_3d_data)
tfce_3d_inplace = zeros(size(Z_3d_data))
flooded_3d_buf = Array{Bool}(undef, size(Z_3d_data))
labeled_3d_buf = Array{Int}(undef, size(Z_3d_data))
tfce_like!(tfce_3d_inplace, Z_3d_data, flooded_3d_buf, labeled_3d_buf)

@assert tfce_3d_result ≈ tfce_3d_inplace "3D TFCE maps don't match"

println("\n3D Benchmarks:")
@time tfce_3d_result = tfce_like(Z_3d_data)
fill!(tfce_3d_inplace, 0.0)
GC.gc()
@time tfce_like!(tfce_3d_inplace, Z_3d_data, flooded_3d_buf, labeled_3d_buf)

println("\nDetailed 3D Benchmarks:")
@btime tfce_like($Z_3d_data; E=0.5, H=2.0, nsteps=50)
fill!(tfce_3d_inplace, 0.0)
GC.gc()
@btime tfce_like!($tfce_3d_inplace, $Z_3d_data, $flooded_3d_buf, $labeled_3d_buf; E=0.5, H=2.0, nsteps=50)

p1 = heatmap(Z_smooth, c=:terrain, title="Smoothed Terrain (Z_smooth)", aspect_ratio=1)
p2 = heatmap(masks[2], c=:nipy_spectral, title="Peaks at level $(round(levels[2]; digits=2)) ($(maximum(masks[2])) clusters)", aspect_ratio=1)
p3 = heatmap(masks[3], c=:nipy_spectral, title="Peaks at level $(round(levels[3]; digits=2)) ($(maximum(masks[3])) clusters)", aspect_ratio=1)
p4 = heatmap(masks[4], c=:nipy_spectral, title="Peaks at level $(round(levels[4]; digits=2)) ($(maximum(masks[4])) clusters)", aspect_ratio=1)
p5 = heatmap(tfce_map, c=:viridis, title="Final TFCE Map", aspect_ratio=1)

plot_2d = plot(p1, p2, p3, p4, p5, layout=(2,3), size=(1000,700))

mid_x = div(nx_3d, 2)
mid_y = div(ny_3d, 2)
mid_z = div(nz_3d, 2)

p3d_1 = heatmap(Z_3d_data[mid_x, :, :], c=:terrain, title="3D Input (x=$mid_x slice)", aspect_ratio=1)
p3d_2 = heatmap(Z_3d_data[:, mid_y, :], c=:terrain, title="3D Input (y=$mid_y slice)", aspect_ratio=1)
p3d_3 = heatmap(Z_3d_data[:, :, mid_z], c=:terrain, title="3D Input (z=$mid_z slice)", aspect_ratio=1)

p3d_4 = heatmap(tfce_3d_result[mid_x, :, :], c=:viridis, title="3D TFCE (x=$mid_x slice)", aspect_ratio=1)
p3d_5 = heatmap(tfce_3d_result[:, mid_y, :], c=:viridis, title="3D TFCE (y=$mid_y slice)", aspect_ratio=1)
p3d_6 = heatmap(tfce_3d_result[:, :, mid_z], c=:viridis, title="3D TFCE (z=$mid_z slice)", aspect_ratio=1)

plot_3d = plot(p3d_1, p3d_2, p3d_3, p3d_4, p3d_5, p3d_6, layout=(2,3), size=(1000,700), title="3D TFCE Visualization")

plot_2d
# plot_3d

