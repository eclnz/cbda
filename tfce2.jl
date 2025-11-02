using LoopVectorization

# ---------------------------
# Union-Find
# ---------------------------
mutable struct UnionFind
    parent::Vector{Int}
    size::Vector{Int}
end

function UnionFind(n::Int)
    parent = collect(1:n)
    size   = ones(Int, n)
    return UnionFind(parent, size)
end

@inline function find!(uf::UnionFind, x::Int)
    p = uf.parent[x]
    if p != x
        uf.parent[x] = find!(uf, p)
    end
    return uf.parent[x]
end

# union that also merges root_contrib and clears the merged root's active_root_flag
@inline function union_with_contrib!(
    uf::UnionFind,
    a::Int,
    b::Int,
    root_contrib::AbstractVector{T},
    active_root_flag::AbstractVector{Bool}
) where T
    ra = find!(uf, a)
    rb = find!(uf, b)
    if ra == rb
        return ra
    end
    # union by size: attach smaller to larger
    if uf.size[ra] < uf.size[rb]
        ra, rb = rb, ra
    end
    uf.parent[rb] = ra
    uf.size[ra]  += uf.size[rb]
    # move contribution from rb -> ra
    root_contrib[ra] += root_contrib[rb]
    root_contrib[rb] = zero(T)
    # rb is no longer an active root
    active_root_flag[rb] = false
    return ra
end

@inline function get_size(uf::UnionFind, x::Int)
    return uf.size[find!(uf, x)]
end

# ---------------------------
# Neighbor precomputation (6-connectivity)
# ---------------------------
function neighbors6(idx::Int, nx::Int, ny::Int, nz::Int)
    # returns neighbor *offsets* for a given linear index
    offsets = Int[]
    k = (idx - 1) ÷ (nx * ny) + 1
    j = ((idx - 1) % (nx * ny)) ÷ nx + 1
    i = ((idx - 1) % (nx * ny)) % nx + 1

    if i > 1;     push!(offsets, -1);      end
    if i < nx;    push!(offsets,  1);      end
    if j > 1;     push!(offsets, -nx);     end
    if j < ny;    push!(offsets,  nx);     end
    if k > 1;     push!(offsets, -nx*ny);  end
    if k < nz;    push!(offsets,  nx*ny);  end
    return offsets
end

function precompute_neighbors6(nx::Int, ny::Int, nz::Int)
    nvox = nx * ny * nz
    neighbor_table = Vector{Vector{Int}}(undef, nvox)
    for idx in 1:nvox
        neighbor_table[idx] = neighbors6(idx, nx, ny, nz)
    end
    return neighbor_table
end

# ---------------------------
# Merge function using precomputed neighbors
# ---------------------------
@inline function merge_active_neighbors!(
    uf::UnionFind,
    active::BitVector,
    idx::Int,
    neighbor_table::Vector{Vector{Int}},
    root_contrib::AbstractVector{T},
    active_root_flag::AbstractVector{Bool}
) where T
    # assume idx is active and currently its parent is itself (or will be found)
    # Merge with each active neighbor, updating contributions and active_root_flag as needed.
    root = find!(uf, idx)
    @inbounds for offset in neighbor_table[idx]
        nidx = idx + offset
        if active[nidx]
            root = union_with_contrib!(uf, root, nidx, root_contrib, active_root_flag)
        end
    end
    return find!(uf, root)
end

# ---------------------------
# Main TFCE (Union-Find incremental) implementation
# ---------------------------
function tfce_3d_unionfind!(
    tfce_out::AbstractArray{T,3},
    t_stat::AbstractArray{T,3};
    E::Float64 = 0.5,
    H::Float64 = 2.0
) where T
    # tfce_out must be same size as t_stat (preallocated) — avoids allocations
    nx, ny, nz = size(t_stat)
    nvox = nx * ny * nz
    flat = reshape(t_stat, nvox)
    tfce_flat = reshape(tfce_out, nvox)  # view into output

    # Data structures - preallocate everything upfront like tfce.jl
    uf = UnionFind(nvox)
    active = falses(nvox)
    neighbor_table = precompute_neighbors6(nx, ny, nz)
    root_contrib = zeros(T, nvox)
    active_root_flag = falses(nvox)
    active_roots = Vector{Int}(undef, nvox)
    temp_active_roots = Vector{Int}(undef, nvox)
    active_roots_len = 0
    temp_active_roots_len = 0

    order = sortperm(flat, rev=true)
    prev_h = flat[order[1]]
    t_min = minimum(flat)

    for idx in order
        h = flat[idx]
        Δh = prev_h - h

        if Δh != 0 && active_roots_len > 0
            temp_active_roots_len = 0
            dh_part = (float(prev_h)^H - float(h)^H)
            @inbounds for i in 1:active_roots_len
                r = active_roots[i]
                if !active_root_flag[r]
                    continue
                end
                actual_r = find!(uf, r)
                if actual_r != r
                    active_root_flag[r] = false
                    continue
                end
                sF = float(uf.size[r])
                contrib = convert(T, (sF^E) * dh_part)
                root_contrib[r] += contrib
                temp_active_roots_len += 1
                temp_active_roots[temp_active_roots_len] = r
            end
            active_roots, temp_active_roots = temp_active_roots, active_roots
            active_roots_len, temp_active_roots_len = temp_active_roots_len, 0
        end

        active[idx] = true
        active_root_flag[idx] = true
        active_roots_len += 1
        active_roots[active_roots_len] = idx

        root = merge_active_neighbors!(uf, active, idx, neighbor_table, root_contrib, active_root_flag)
        root = find!(uf, root)
        active_root_flag[root] = true
        prev_h = h
    end

    if active_roots_len > 0
        dh_part = (float(prev_h)^H - float(t_min)^H)
        temp_active_roots_len = 0
        @inbounds for i in 1:active_roots_len
            r = active_roots[i]
            if !active_root_flag[r]
                continue
            end
            actual_r = find!(uf, r)
            if actual_r != r
                active_root_flag[r] = false
                continue
            end
            sF = float(uf.size[r])
            root_contrib[r] += convert(T, (sF^E) * dh_part)
            temp_active_roots_len += 1
            temp_active_roots[temp_active_roots_len] = r
        end
        active_roots, temp_active_roots = temp_active_roots, active_roots
        active_roots_len, temp_active_roots_len = temp_active_roots_len, 0
    end

    # Propagate root_contrib to all voxels
    @inbounds for i in 1:nvox
        tfce_flat[i] = root_contrib[find!(uf, i)]
    end

    return tfce_out
end

function tfce_3d_unionfind!(
    tfce_out::AbstractArray{T,3},
    t_stat::AbstractArray{T,3},
    uf::UnionFind,
    active::BitVector,
    neighbor_table::Vector{Vector{Int}},
    root_contrib::AbstractVector{T},
    active_root_flag::BitVector,
    active_roots::Vector{Int},
    temp_active_roots::Vector{Int},
    order::Vector{Int},
    active_roots_len::Ref{Int},
    temp_active_roots_len::Ref{Int};
    E::Float64 = 0.5,
    H::Float64 = 2.0
) where T
    nx, ny, nz = size(t_stat)
    nvox = nx * ny * nz
    flat = reshape(t_stat, nvox)
    tfce_flat = reshape(tfce_out, nvox)

    fill!(active, false)
    fill!(root_contrib, zero(T))
    fill!(active_root_flag, false)
    active_roots_len = 0
    temp_active_roots_len = 0

    for i in 1:length(uf.parent)
        uf.parent[i] = i
        uf.size[i] = 1
    end

    order_result = sortperm(flat, rev=true)
    resize!(order, length(order_result))
    copyto!(order, order_result)
    prev_h = flat[order[1]]
    t_min = minimum(flat)

    for idx in order
        h = flat[idx]
        Δh = prev_h - h

        if Δh != 0 && active_roots_len[] > 0
            temp_active_roots_len[] = 0
            dh_part = (float(prev_h)^H - float(h)^H)
            @inbounds for i in 1:active_roots_len[]
                r = active_roots[i]
                if !active_root_flag[r]
                    continue
                end
                actual_r = find!(uf, r)
                if actual_r != r
                    active_root_flag[r] = false
                    continue
                end
                sF = float(uf.size[r])
                contrib = convert(T, (sF^E) * dh_part)
                root_contrib[r] += contrib
                temp_active_roots_len[] += 1
                temp_active_roots[temp_active_roots_len[]] = r
            end
            active_roots, temp_active_roots = temp_active_roots, active_roots
            active_roots_len[], temp_active_roots_len[] = temp_active_roots_len[], 0
        end

        active[idx] = true
        active_root_flag[idx] = true
        active_roots_len[] += 1
        active_roots[active_roots_len[]] = idx

        root = merge_active_neighbors!(uf, active, idx, neighbor_table, root_contrib, active_root_flag)
        root = find!(uf, root)
        active_root_flag[root] = true
        prev_h = h
    end

    if active_roots_len[] > 0
        dh_part = (float(prev_h)^H - float(t_min)^H)
        temp_active_roots_len[] = 0
        @inbounds for i in 1:active_roots_len[]
            r = active_roots[i]
            if !active_root_flag[r]
                continue
            end
            actual_r = find!(uf, r)
            if actual_r != r
                active_root_flag[r] = false
                continue
            end
            sF = float(uf.size[r])
            root_contrib[r] += convert(T, (sF^E) * dh_part)
            temp_active_roots_len[] += 1
            temp_active_roots[temp_active_roots_len[]] = r
        end
        active_roots, temp_active_roots = temp_active_roots, active_roots
        active_roots_len[], temp_active_roots_len[] = temp_active_roots_len[], 0
    end

    @inbounds for i in 1:nvox
        tfce_flat[i] = root_contrib[find!(uf, i)]
    end

    return tfce_out
end

function tfce_3d_unionfind(t_stat::Array{T,3}; E::Float64 = 0.5, H::Float64 = 2.0) where T
    tfce_out = zeros(T, size(t_stat))
    return tfce_3d_unionfind!(tfce_out, t_stat; E=E, H=H)
end
