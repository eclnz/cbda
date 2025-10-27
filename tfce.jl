using LinearAlgebra
using ProfileSVG
using Profile

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

# --------------------------
# Example usage and profiling
x, y, z = 100, 100, 100
t_stat = rand(Float32, x, y, z) .* 3

println("Computing TFCE on t-stat image...")
unique_heights = sort(unique(t_stat), rev=true)
println(length(unique_heights))
@time tfce_result = tfce_3d(t_stat, E=1.0, H=2.0)
@time tfce_result = tfce_3d(t_stat, E=1.0, H=2.0)
nothing
Profile.clear()
@profile tfce_result = tfce_3d(t_stat, E=1.0, H=2.0)
ProfileSVG.save("tfce_profile.svg"; width=5000, height=1200)
println("ProfileSVG output saved to tfce_profile.svg")
println("\nProfile text output:")
Profile.print()

# 1.527334 seconds (11.49 M allocations: 7.224 GiB, 5.89% gc time, 2.22% compilation time)
# 1.397978 seconds (11.45 M allocations: 7.223 GiB, 5.40% gc time)

# 0.307403 seconds (224.20 k allocations: 16.385 MiB, 1.33% gc time, 96.47% compilation time)
# 0.009021 seconds (14.46 k allocations: 7.426 MiB, 26.16% gc time)

nothing