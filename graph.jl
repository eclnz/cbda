graph.jl

# -------------------- Graph Tree Structure --------------------
mutable struct GraphNode
    id::Int
    left_i_top::Int
    right_i_top::Int
    left_i_bottom::Int
    right_i_bottom::Int
    height::Float64
    box_area::Float64
    side_area::Float64
    area::Float64
    cumul_area::Float64
    parent::Union{GraphNode, Nothing}
    children::Vector{GraphNode}
end

GraphNode(id::Int, left::Int, right::Int, height::Float64) = 
    GraphNode(id, left, right, left, right, height, 0.0, 0.0, 0.0, 0.0, nothing, GraphNode[])

function find_neighbors_at_level(min_idx::Int, candidate_indices::Vector{Int}, signal::Vector{Float64}, top_basin_bounds::Dict{Int, Tuple{Int, Int, Float64}})
    my_left, my_right, _ = top_basin_bounds[min_idx]
    left_neighbor = nothing
    right_neighbor = nothing
    
    for idx in candidate_indices
        idx == min_idx && continue
        cand_left, cand_right, _ = top_basin_bounds[idx]
        
        overlaps = !(cand_right < my_left || cand_left > my_right)
        
        if overlaps
            if idx < min_idx && (left_neighbor === nothing || signal[idx] < signal[left_neighbor])
                left_neighbor = idx
            elseif idx > min_idx && (right_neighbor === nothing || signal[idx] < signal[right_neighbor])
                right_neighbor = idx
            end
        end
    end
    
    return left_neighbor, right_neighbor
end

function find_parent(min_idx::Int, sorted_minima::Vector{Int}, processed::Set{Int}, signal::Vector{Float64}, top_basin_bounds::Dict{Int, Tuple{Int, Int, Float64}}, nodes::Dict{Int,GraphNode})
    my_left, my_right, _ = top_basin_bounds[min_idx]
    parent_node = nothing
    best_parent_height = -Inf
    for cand_idx in sorted_minima
        if !(cand_idx in processed) || cand_idx == min_idx
            continue
        end
        cand_left, cand_right, _ = top_basin_bounds[cand_idx]
        contains = (cand_left <= my_left && cand_right >= my_right)
        if contains && signal[cand_idx] < signal[min_idx] && signal[cand_idx] > best_parent_height
            parent_node = nodes[cand_idx]
            best_parent_height = signal[cand_idx]
        end
    end
    return parent_node
end

function compute_bottom_bounds(child::GraphNode, parent::GraphNode)
    if child.id < parent.id
        left_bottom = parent.left_i_top
        right_bottom = parent.id
    else
        left_bottom = parent.id
        right_bottom = parent.right_i_top
    end
    return left_bottom, right_bottom

end

function build_graph_tree(minima_indices::Vector{Int}, top_basin_bounds::Dict{Int, Tuple{Int, Int, Float64}}, signal::Vector{Float64})
    nodes = Dict{Int, GraphNode}()
    for min_idx in minima_indices
        l, r, b = top_basin_bounds[min_idx]
        nodes[min_idx] = GraphNode(min_idx, l, r, b)
    end

    sorted_minima = sort(minima_indices, by=x->signal[x])
    root = nodes[sorted_minima[1]]
    processed = Set{Int}()
    
    for (iter, min_idx) in enumerate(sorted_minima)
        if iter == 1
            push!(processed, min_idx)
            continue
        end
        parent_node = find_parent(min_idx, sorted_minima, processed, signal, top_basin_bounds, nodes)
        if parent_node !== nothing
            nodes[min_idx].parent = parent_node
            push!(parent_node.children, nodes[min_idx])
            child_node = nodes[min_idx]
            lb, rb = compute_bottom_bounds(child_node, parent_node)
            child_node.left_i_bottom = lb
            child_node.right_i_bottom = rb
            println("Node $min_idx (height=$(round(signal[min_idx], digits=3))) -> parent $(parent_node.id) (height=$(round(parent_node.height, digits=3)))")
        end
        push!(processed, min_idx)
    end
    
    return root, nodes
end

# -------------------- Node Utilities --------------------
node_area_above(ws::WSData, pos::Int, bottom::Float64) =
    ws.areas[pos] + (ws.height[pos] - bottom) * (ws.right_bounds[pos] - ws.left_bounds[pos] + 1)

function overlapping_bottoms(ws::WSData, positions::Vector{Int}, fallback_pos::Int)
    # Optimized: consider only baselines lower than current positions
    sorted_nodes = sortperm(ws.height)
    bottoms = Float64[]
    for pos in positions
        l, r = ws.left_bounds[pos], ws.right_bounds[pos]
        hb = maximum([ws.height[c] for c in sorted_nodes if ws.height[c] < ws.height[pos] &&
                     !(ws.right_bounds[c] < l || ws.left_bounds[c] > r)] ; init=ws.height[fallback_pos])
        push!(bottoms, hb)
    end
    bottoms
end