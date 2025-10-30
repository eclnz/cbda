using LinearAlgebra
using ProfileSVG
using Profile
using LoopVectorization
using ImageFiltering
using BenchmarkTools
using Plots
using Random

@inline function linear_idx(i, j, k, nx, ny, nz)
    return i + (j - 1) * nx + (k - 1) * nx * ny
end

@inline function add_neighbors_to_frontier!(
    frontier::Vector{Int}, frontier_len::Int, visited::Vector{Bool}, stat::Vector{T}, h::T, sign::Int,
    cidx::Int, nx::Int, ny::Int, nz::Int
) where T
    k = (cidx - 1) ÷ (nx * ny) + 1
    j = ((cidx - 1) % (nx * ny)) ÷ nx + 1
    i = ((cidx - 1) % (nx * ny)) % nx + 1
    
    for (offset, in_bounds) in ((-1, i > 1), (1, i < nx), (-nx, j > 1), (nx, j < ny), (-nx*ny, k > 1), (nx*ny, k < nz))
        if in_bounds
            nidx = cidx + offset
            @inbounds if !visited[nidx] && sign * stat[nidx] >= h
                visited[nidx] = true
                frontier_len += 1
                frontier[frontier_len] = nidx
            end
        end
    end
    return frontier_len
end

function tfce_threshold_level!(tfce::Vector{T}, stat::Vector{T}, visited::Vector{Bool}, 
                      frontier::Vector{Int}, cluster::Vector{Int}, h::T, t_min::T, 
                      E::Float64, H::Float64, sign::Int, nx::Int, ny::Int, nz::Int) where T
    height_part = (h - t_min)^H
    
    fill!(visited, false)
    
    for idx in 1:length(stat)
        @inbounds val = sign * stat[idx]
        if val < h || visited[idx]
            continue
        end
        
        frontier_len = 0
        cluster_len = 0
        
        visited[idx] = true
        frontier_len += 1
        frontier[frontier_len] = idx
        
        while frontier_len > 0
            cidx = frontier[frontier_len]
            frontier_len -= 1
            
            cluster_len += 1
            cluster[cluster_len] = cidx
            
            frontier_len = add_neighbors_to_frontier!(frontier, frontier_len, visited, stat, h, sign, cidx, nx, ny, nz)
        end
        
        if cluster_len > 0
            contribution = (cluster_len^E) * height_part
            @inbounds for i in 1:cluster_len
                cidx = cluster[i]
                tfce[cidx] += sign * contribution
            end
        end
    end
end

function tfce_3d(t_stat::Array{T,3}; E=0.5, H=2.0, dh=0.1, calc_neg=true) where T
    dims = size(t_stat)
    nx, ny, nz = dims
    t_stat_flat = reshape(t_stat, length(t_stat))
    tfce_flat = zeros(T, length(t_stat))
    t_min = convert(T, minimum(t_stat))
    t_max = convert(T, maximum(abs.(t_stat_flat)))
    
    dh_T = convert(T, dh)
    thresholds = dh_T:dh_T:t_max
    
    visited = zeros(Bool, nx*ny*nz)
    max_voxels = nx * ny * nz
    frontier = Vector{Int}(undef, max_voxels)
    cluster = Vector{Int}(undef, max_voxels)
    
    for h in thresholds
        tfce_threshold_level!(tfce_flat, t_stat_flat, visited, frontier, cluster, h, t_min, E, H, 1, nx, ny, nz)
    end
    
    if calc_neg
        for h in thresholds
            tfce_threshold_level!(tfce_flat, t_stat_flat, visited, frontier, cluster, h, t_min, E, H, -1, nx, ny, nz)
        end
    end
    
    return reshape(tfce_flat, dims)
end

function kernel_3d_gradient(volume::Array{T,3}) where T
    gx, gy, gz = ImageFiltering.imgradients(volume, ImageFiltering.KernelFactors.ando3, "replicate")
    return sqrt.(gx .* gx .+ gy .* gy .+ gz .* gz)
end

function kernel_3d_gradient2(volume::Array{T,3}) where T
    static_kernel = ImageFiltering.KernelFactors.ando3
    gx, gy, gz = imgradients(volume, static_kernel, "replicate")

    out = similar(volume)

    @turbo for i in eachindex(out)
        out[i] = sqrt(gx[i]^2 + gy[i]^2 + gz[i]^2)
    end

    return out
end

 

function local_extrema_coords(vol::Array{T,3}; exclude_border=true) where T
    nx, ny, nz = size(vol)
    minima = Vector{CartesianIndex{3}}()
    maxima = Vector{CartesianIndex{3}}()
    
    xstart = exclude_border ? 2 : 1
    xend = exclude_border ? (nx-1) : nx
    ystart = exclude_border ? 2 : 1
    yend = exclude_border ? (ny-1) : ny
    zstart = exclude_border ? 2 : 1
    zend = exclude_border ? (nz-1) : nz
    
    for k in zstart:zend, j in ystart:yend, i in xstart:xend
        val = vol[i, j, k]
        is_min = true
        is_max = true
        for dk in -1:1, dj in -1:1, di in -1:1
            (di == 0 && dj == 0 && dk == 0) && continue
            ni, nj, nk = i + di, j + dj, k + dk
            if 1 <= ni <= nx && 1 <= nj <= ny && 1 <= nk <= nz
                nval = vol[ni, nj, nk]
                if nval <= val
                    is_min = false
                end
                if nval >= val
                    is_max = false
                end
                !is_min && !is_max && break
            end
        end
        if is_min
            push!(minima, CartesianIndex(i, j, k))
        elseif is_max
            push!(maxima, CartesianIndex(i, j, k))
        end
    end
    return minima, maxima
end

function gradient_zero_coords(g::Array{T,3}; eps=1e-3, step=nothing, exclude_border=true) where T
    if isnothing(step)
        step = (maximum(g) - minimum(g)) / 1000
    end
    mask = round.(Int, g ./ step) .== 0
    candidates = [idx for idx in findall(mask) if g[idx] <= convert(T, eps)]
    if exclude_border
        nx, ny, nz = size(g)
        verified = [idx for idx in candidates if 1 < idx[1] < nx && 1 < idx[2] < ny && 1 < idx[3] < nz]
        return verified
    end
    return candidates
end

function gradient_magnitude_3d_fd!(out, volume)
    nx, ny, nz = size(volume)
    @turbo for k in 1:nz, j in 1:ny, i in 1:nx
        ip = (i % nx) + 1
        im = ((i - 2 + nx) % nx) + 1
        jp = (j % ny) + 1
        jm = ((j - 2 + ny) % ny) + 1
        kp = (k % nz) + 1
        km = ((k - 2 + nz) % nz) + 1
        gx = (volume[ip, j, k] - volume[im, j, k]) * 0.5
        gy = (volume[i, jp, k] - volume[i, jm, k]) * 0.5
        gz = (volume[i, j, kp] - volume[i, j, km]) * 0.5
        out[i, j, k] = sqrt(gx^2 + gy^2 + gz^2)
    end
    return out
end

# --------------------------
# Example usage and profiling
x, y, z = 100, 100, 100
Random.seed!(53)
t_stat = rand(Float32, x, y, z) .* 3

println("Computing 3D kernel gradient...")
σ = 6.0f0
gk = ImageFiltering.KernelFactors.IIRGaussian((σ, σ, σ))
t_smooth = imfilter(t_stat, gk, "replicate")
@time grad_result = kernel_3d_gradient(t_smooth)
@time grad_result2 = kernel_3d_gradient(t_smooth)

@time grad_result = kernel_3d_gradient2(t_smooth)
@time grad_result = kernel_3d_gradient2(t_smooth)

@time grad_result = gradient_magnitude_3d_fd!(similar(t_smooth), t_smooth)
@time grad_result = gradient_magnitude_3d_fd!(similar(t_smooth), t_smooth)

g1 = kernel_3d_gradient(t_smooth)
g2 = kernel_3d_gradient2(t_smooth)
g3 = gradient_magnitude_3d_fd!(similar(t_smooth), t_smooth)
rx = 3:(x-2); ry = 3:(y-2); zmid = clamp(fld(z, 2), 3, z-2)
s0 = @view t_smooth[rx, ry, zmid]
s1 = @view g1[rx, ry, zmid]
s2 = @view g2[rx, ry, zmid]
s3 = @view g3[rx, ry, zmid]
mn = minimum((minimum(s1), minimum(s2), minimum(s3)))
mx = maximum((maximum(s1), maximum(s2), maximum(s3)))
p1 = heatmap(s0, title="t_stat (smoothed)")
p2 = heatmap(s1, title="imgradients", clim=(mn, mx))
p3 = heatmap(s2, title="imgradients2", clim=(mn, mx))
p4 = heatmap(s3, title="fd!", clim=(mn, mx))
p = plot(p1, p2, p3, p4, layout=(2,2), size=(1400,900))
display(p)


println("Computing TFCE on t-stat image...")
unique_heights = sort(unique(t_stat), rev=true)
println(length(unique_heights))
@time tfce_result = tfce_3d(t_stat, E=1.0, H=2.0)
@time tfce_result = tfce_3d(t_stat, E=1.0, H=2.0)
nothing
println("Fast local extrema detection (no gradient computation):")
@time minima_fast, maxima_fast = local_extrema_coords(t_smooth; exclude_border=true)
println("  Found $(length(minima_fast)) minima and $(length(maxima_fast)) maxima")
zeros_idx = vcat(minima_fast, maxima_fast)
if !isempty(zeros_idx)
    center_min = zeros_idx[(length(zeros_idx) + 1) ÷ 2]
    xmid, ymid, zmid = center_min[1], center_min[2], center_min[3]
    center_val = grad_result2[center_min]
    println("Center point $(center_min): g = $(center_val)")
    
    xy_points = filter(idx -> idx[3] == zmid, zeros_idx)
    xz_points = filter(idx -> idx[2] == ymid, zeros_idx)
    yz_points = filter(idx -> idx[1] == xmid, zeros_idx)
    
    p1 = heatmap(grad_result2[:, :, zmid], title="XY plane at z=$(zmid)", xlabel="x", ylabel="y")
    if !isempty(xy_points)
        xs = [idx[1] for idx in xy_points]
        ys = [idx[2] for idx in xy_points]
        scatter!(p1, xs, ys; m=:circle, ms=4, c=:yellow, lab="g≈0", linewidth=1.5)
    end
    
    nx, ny, nz = size(grad_result2)
    xz_slice = [grad_result2[i, ymid, k] for i in 1:nx, k in 1:nz]
    p2 = heatmap(xz_slice, title="XZ plane at y=$(ymid)", xlabel="x", ylabel="z", aspect_ratio=:auto)
    if !isempty(xz_points)
        xs = [idx[1] for idx in xz_points]
        zs = [idx[3] for idx in xz_points]
        scatter!(p2, xs, zs; m=:circle, ms=4, c=:yellow, lab="g≈0", linewidth=1.5)
    end
    
    yz_slice = [grad_result2[xmid, j, k] for j in 1:ny, k in 1:nz]
    p3 = heatmap(yz_slice, title="YZ plane at x=$(xmid)", xlabel="y", ylabel="z", aspect_ratio=:auto)
    if !isempty(yz_points)
        ys = [idx[2] for idx in yz_points]
        zs = [idx[3] for idx in yz_points]
        scatter!(p3, ys, zs; m=:circle, ms=4, c=:yellow, lab="g≈0", linewidth=1.5)
    end
    
    p = plot(p1, p2, p3, layout=(1,3), size=(1800,600))
    display(p)
else
    println("No gradient zero points found")
end
Profile.clear()
@profile tfce_result = tfce_3d(t_stat, E=1.0, H=2.0)
ProfileSVG.save("tfce_profile.svg"; width=5000, height=1200)
println("ProfileSVG output saved to tfce_profile.svg")
println("\nProfile text output:")
# Profile.print()

# 1.527334 seconds (11.49 M allocations: 7.224 GiB, 5.89% gc time, 2.22% compilation time)
# 1.397978 seconds (11.45 M allocations: 7.223 GiB, 5.40% gc time)

# 0.307403 seconds (224.20 k allocations: 16.385 MiB, 1.33% gc time, 96.47% compilation time)
# 0.009021 seconds (14.46 k allocations: 7.426 MiB, 26.16% gc time)

nothing