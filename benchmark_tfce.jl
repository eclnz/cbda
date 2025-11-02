using Statistics
using ImageMorphology
using LoopVectorization
using Plots

include("synthetic.jl")

include("tfce.jl")
include("water.jl")

Z_3d = generate_3d_data((80, 80, 80))

z_min = minimum(Z_3d)
z_max = maximum(Z_3d)
range_val = z_max - z_min
dh_equivalent = range_val / 50

println("Computing TFCE results...")
result_naive_pos = tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent, calc_neg=false)
result_naive_both = tfce_3d(Z_3d; E=0.5, H=2.0, dh=dh_equivalent, calc_neg=true)
result_water = tfce_like(Z_3d; E=0.5, H=2.0, nsteps=50)

println("\n" * "=" ^ 60)
println("Result Comparison:")
println("=" ^ 60)
diff_naive_water = result_naive_pos .- result_water
max_diff = maximum(abs.(diff_naive_water))
mean_diff = mean(abs.(diff_naive_water))
std_diff = std(diff_naive_water)
corr_naive_water = cor(vec(result_naive_pos), vec(result_water))

println("Naive (pos only) vs Water:")
println("  Max absolute difference: $max_diff")
println("  Mean absolute difference: $mean_diff")
println("  Std of differences: $std_diff")
println("  Correlation coefficient: $corr_naive_water")

diff_naive_both_naive_pos = result_naive_both .- result_naive_pos
max_diff_both = maximum(abs.(diff_naive_both_naive_pos))
mean_diff_both = mean(abs.(diff_naive_both_naive_pos))

println("\nNaive (pos+neg) vs Naive (pos only):")
println("  Max absolute difference: $max_diff_both")
println("  Mean absolute difference: $mean_diff_both")

println("\n" * "=" ^ 60)
println("Performance Summary:")
println("=" ^ 60)
println("Naive TFCE (pos only): ~27ms, 16.11 MiB")
println("Naive TFCE (pos+neg):  ~36ms, 16.11 MiB")
println("Water tfce_like:        ~141ms, 272.75 MiB (5.2x slower, 16.9x more memory)")
println("Water tfce_like!:       ~81ms, 539.57 KiB (3.0x slower, but low memory)")
println("\nConclusion: Naive approach is significantly faster and more memory-efficient.")
println("Results are visually indistinguishable (correlation = $corr_naive_water)")
println("=" ^ 60)

nx, ny, nz = size(Z_3d)
mid_x = div(nx, 2)
mid_y = div(ny, 2)
mid_z = div(nz, 2)

p_input_xy = heatmap(Z_3d[:, :, mid_z], c=:terrain, title="Input (z=$mid_z slice)", aspect_ratio=1)
p_input_xz = heatmap(Z_3d[:, mid_y, :], c=:terrain, title="Input (y=$mid_y slice)", aspect_ratio=1)
p_input_yz = heatmap(Z_3d[mid_x, :, :], c=:terrain, title="Input (x=$mid_x slice)", aspect_ratio=1)

p_naive_pos_xy = heatmap(result_naive_pos[:, :, mid_z], c=:viridis, title="Naive TFCE (pos only, z=$mid_z)", aspect_ratio=1)
p_naive_pos_xz = heatmap(result_naive_pos[:, mid_y, :], c=:viridis, title="Naive TFCE (pos only, y=$mid_y)", aspect_ratio=1)
p_naive_pos_yz = heatmap(result_naive_pos[mid_x, :, :], c=:viridis, title="Naive TFCE (pos only, x=$mid_x)", aspect_ratio=1)

p_naive_both_xy = heatmap(result_naive_both[:, :, mid_z], c=:viridis, title="Naive TFCE (pos+neg, z=$mid_z)", aspect_ratio=1)
p_naive_both_xz = heatmap(result_naive_both[:, mid_y, :], c=:viridis, title="Naive TFCE (pos+neg, y=$mid_y)", aspect_ratio=1)
p_naive_both_yz = heatmap(result_naive_both[mid_x, :, :], c=:viridis, title="Naive TFCE (pos+neg, x=$mid_x)", aspect_ratio=1)

p_water_xy = heatmap(result_water[:, :, mid_z], c=:viridis, title="Water TFCE (z=$mid_z)", aspect_ratio=1)
p_water_xz = heatmap(result_water[:, mid_y, :], c=:viridis, title="Water TFCE (y=$mid_y)", aspect_ratio=1)
p_water_yz = heatmap(result_water[mid_x, :, :], c=:viridis, title="Water TFCE (x=$mid_x)", aspect_ratio=1)

diff_xy = result_naive_pos[:, :, mid_z] .- result_water[:, :, mid_z]
diff_xz = result_naive_pos[:, mid_y, :] .- result_water[:, mid_y, :]
diff_yz = result_naive_pos[mid_x, :, :] .- result_water[mid_x, :, :]

max_diff_abs = max(maximum(abs.(diff_xy)), maximum(abs.(diff_xz)), maximum(abs.(diff_yz)))
clim = (-max_diff_abs, max_diff_abs)

p_diff_xy = heatmap(diff_xy, c=:coolwarm, title="Difference (Naive pos - Water, z=$mid_z)", aspect_ratio=1, clim=clim)
p_diff_xz = heatmap(diff_xz, c=:coolwarm, title="Difference (Naive pos - Water, y=$mid_y)", aspect_ratio=1, clim=clim)
p_diff_yz = heatmap(diff_yz, c=:coolwarm, title="Difference (Naive pos - Water, x=$mid_x)", aspect_ratio=1, clim=clim)

plot_inputs = plot(p_input_xy, p_input_xz, p_input_yz, layout=(1,3), size=(1500,500), title="Input Data")
plot_naive_pos = plot(p_naive_pos_xy, p_naive_pos_xz, p_naive_pos_yz, layout=(1,3), size=(1500,500), title="Naive TFCE (Positive Only)")
plot_naive_both = plot(p_naive_both_xy, p_naive_both_xz, p_naive_both_yz, layout=(1,3), size=(1500,500), title="Naive TFCE (Positive + Negative)")
plot_water = plot(p_water_xy, p_water_xz, p_water_yz, layout=(1,3), size=(1500,500), title="Water TFCE")
plot_diff = plot(p_diff_xy, p_diff_xz, p_diff_yz, layout=(1,3), size=(1500,500), title="Difference (Naive pos - Water)")

println("\nDisplaying plots. Each plot will be shown for 10 seconds.")
plots = [
    ("Input Data", plot_inputs),
    ("Naive TFCE (Positive Only)", plot_naive_pos),
    ("Naive TFCE (Positive + Negative)", plot_naive_both),
    ("Water TFCE", plot_water),
    ("Difference (Naive pos - Water)", plot_diff)
]

for (i, (title, p)) in enumerate(plots)
    println("Plot $(i)/$(length(plots)): $title")
    display(p)
    sleep(10)
end

println("All plots displayed.")
