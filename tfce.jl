using LoopVectorization

function add_neighbors_to_frontier!(
    frontier::AbstractVector{Int}, frontier_len::Int, visited::AbstractVector{Bool}, stat::AbstractVector{T}, h::T,
    cidx::Int, nx::Int, ny::Int, nz::Int
) where T
    k = (cidx - 1) ÷ (nx * ny) + 1
    j = ((cidx - 1) % (nx * ny)) ÷ nx + 1
    i = ((cidx - 1) % (nx * ny)) % nx + 1
    
    for (offset, in_bounds) in ((-1, i > 1), (1, i < nx), (-nx, j > 1), (nx, j < ny), (-nx*ny, k > 1), (nx*ny, k < nz))
        if in_bounds
            nidx = cidx + offset
            @inbounds if !visited[nidx] && stat[nidx] >= h
                visited[nidx] = true
                frontier_len += 1
                frontier[frontier_len] = nidx
            end
        end
    end
    return frontier_len
end

function tfce_threshold_level!(tfce::AbstractVector{T}, stat::AbstractVector{T}, visited::AbstractVector{Bool}, 
                      frontier::AbstractVector{Int}, cluster::AbstractVector{Int}, h::T, t_min::T, 
                      E::Float64, H::Float64, nx::Int, ny::Int, nz::Int) where T
    height_part = (h - t_min)^H
    
    fill!(visited, false)
    
    for idx in eachindex(stat)
        if stat[idx] < h || visited[idx]
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
            
            frontier_len = add_neighbors_to_frontier!(frontier, frontier_len, visited, stat, h, cidx, nx, ny, nz)
        end
        
        if cluster_len > 0
            contribution = (cluster_len^E) * height_part
            for i in 1:cluster_len
                cidx = cluster[i]
                tfce[cidx] += contribution
            end
        end
    end
end

function tfce_3d(t_stat::Array{T,3}; E=0.5, H=2.0, dh=0.1) where T
    dims = size(t_stat)
    nx, ny, nz = dims
    t_stat_flat = reshape(t_stat, length(t_stat))
    tfce_flat = zeros(T, length(t_stat))
    t_min = convert(T, minimum(t_stat))
    t_max = convert(T, maximum(t_stat_flat))
    
    dh_T = convert(T, dh)
    thresholds = dh_T:dh_T:t_max
    
    visited = zeros(Bool, nx*ny*nz)
    max_voxels = nx * ny * nz
    frontier = Vector{Int}(undef, max_voxels)
    cluster = Vector{Int}(undef, max_voxels)
    
    for h in thresholds
        tfce_threshold_level!(tfce_flat, t_stat_flat, visited, frontier, cluster, h, t_min, E, H, nx, ny, nz)
    end
    
    return reshape(tfce_flat, dims)
end