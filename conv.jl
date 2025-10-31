using Random
using Plots

# -------------------- Constants --------------------
const SEED = 102
const N_POINTS = 120
const KERNEL_SIZE = 21
const KERNEL_SIGMA = 2.5
const ADD_OFFSET = true
const OFFSET_VALUE = 0.1
const DISCRETE_LEVELS = 20000
const PLOT_MAX_RECTANGLES = 100

# -------------------- Signal Utilities --------------------
Random.seed!(SEED)

function gaussian_kernel(size::Int, sigma::Float64)
    x = range(-div(size, 2), stop=div(size, 2), length=size)
    k = exp.(-x.^2 ./ (2*sigma^2))
    k ./ sum(k)
end

function conv_valid(signal::Vector{Float64}, kernel::Vector{Float64})
    pad = div(length(kernel), 2)
    padded = vcat(fill(signal[1], pad), signal, fill(signal[end], pad))
    n = length(padded) - length(kernel) + 1
    result = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        window = @view padded[i:i+length(kernel)-1]
        result[i] = sum(window .* kernel)
    end
    result
end

function discretize_amplitude(signal::Vector{Float64}, levels::Int)
    min_val, max_val = minimum(signal), maximum(signal)
    range_val = max_val - min_val
    normalized = (signal .- min_val) ./ range_val
    discretized = floor.(normalized .* levels) ./ levels .* range_val .+ min_val
    discretized
end

function process_signal(signal::Vector{Float64})
    kernel = gaussian_kernel(KERNEL_SIZE, KERNEL_SIGMA)
    smooth_signal = conv_valid(signal, kernel)
    smooth_signal .+= abs(minimum(smooth_signal)) .+ OFFSET_VALUE
    discretize_amplitude(smooth_signal, DISCRETE_LEVELS)
end

function find_extrema_indices(signal::Vector{Float64}, cmp_prev::Function, cmp_next::Function)
    indices = Int[]
    @inbounds for idx in 2:(length(signal)-1)
        previous = signal[idx-1]
        current = signal[idx]
        next = signal[idx+1]
        if cmp_prev(current, previous) && cmp_next(current, next)
            push!(indices, idx)
        end
    end
    return indices
end

find_minima_indices(signal::Vector{Float64}) = find_extrema_indices(signal, <=, <)
find_maxima_indices(signal::Vector{Float64}) = find_extrema_indices(signal, >=, >)

# -------------------- Ray Projection --------------------
function project_ray(signal::Vector{Float64}, idx::Int, target_val::Float64, dir::Int)
    n = length(signal)
    i = idx + dir
    while 1 ≤ i ≤ n
        if signal[i] < target_val
            return i
        end
        i += dir
    end
    dir == -1 ? 1 : n
end

project_ray_left(signal, idx, target_val) = project_ray(signal, idx, target_val, -1)
project_ray_right(signal, idx, target_val) = project_ray(signal, idx, target_val, 1)

function get_top_basin_bounds(signal::Vector{Float64}, minima_indices::Vector{Int})
    top_basin_bounds = Dict{Int,Tuple{Int,Int,Float64}}()
    for min_idx in minima_indices
        val = signal[min_idx]
        left_proj = project_ray_left(signal, min_idx, val)
        right_proj = project_ray_right(signal, min_idx, val)
        top_basin_bounds[min_idx] = (left_proj, right_proj, val)
    end
    return top_basin_bounds
end

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

function get_node_width(node::GraphNode)
    return node.right_i_top - node.left_i_top + 1
end


function get_left_area(node::GraphNode, signal::Vector{Float64})
    if node.parent === nothing
        return 0.0
    end
    parent_baseline = node.parent.height

    left_area = 0.0
    if node.left_i_top != node.left_i_bottom
        lstart = min(node.left_i_top, node.left_i_bottom)
        lend = max(node.left_i_top, node.left_i_bottom) - 1
        if lstart <= lend
            for i in lstart:lend
                clamped = max(min(signal[i], node.height), parent_baseline)
                left_area += clamped - parent_baseline
            end
        end
    end
    return left_area
end

function get_right_area(node::GraphNode, signal::Vector{Float64})
    if node.parent === nothing
        return 0.0
    end
    parent_baseline = node.parent.height

    right_area = 0.0
    if node.right_i_top != node.right_i_bottom
        rstart = min(node.right_i_top, node.right_i_bottom) + 1
        rend = max(node.right_i_top, node.right_i_bottom)
        if rstart <= rend
            for i in rstart:rend
                clamped = max(min(signal[i], node.height), parent_baseline)
                right_area += clamped - parent_baseline
            end
        end
    end
    return right_area
end

function get_side_area(node::GraphNode, signal::Vector{Float64})
    return get_left_area(node, signal) + get_right_area(node, signal)
end

function set_graph_areas!(node::GraphNode, signal::Vector{Float64})
    width = get_node_width(node)
    baseline_area = node.height * width
    
    if node.parent === nothing
        node.box_area = baseline_area
        node.side_area = 0.0
        node.area = baseline_area
    else
        box_height = node.height - node.parent.height
        node.box_area = width * box_height
        node.side_area = get_side_area(node, signal)
        node.area = node.box_area + node.side_area
    end
    
    node.cumul_area = node.area
    for child in node.children
        set_graph_areas!(child, signal)
        node.cumul_area += child.cumul_area
    end
    
    if node.parent === nothing
        println("Node $(node.id): width=$width, height=$(round(node.height, digits=3)), area=$(round(node.area, digits=2)), cumul=$(round(node.cumul_area, digits=2))")
    else
        box_height = node.height - node.parent.height
        println("Node $(node.id): width=$width, box_h=$(round(box_height, digits=3)), box=$(round(node.box_area, digits=2)), side=$(round(node.side_area, digits=2)), area=$(round(node.area, digits=2)), cumul=$(round(node.cumul_area, digits=2))")
    end
end

function print_tree(node::GraphNode, indent::Int=0)
    prefix = "  " ^ indent
    parent_id = node.parent === nothing ? "ROOT" : string(node.parent.id)
    # println("$(prefix)Node $(node.id): height=$(round(node.height, digits=3)), bounds=[$(node.left_bound), $(node.right_bound)], parent=$(parent_id), children=$(length(node.children))")
    # for child in node.children
    #     print_tree(child, indent + 1)
    # end
end

# -------------------- WSData --------------------
struct WSData
    node_ids::Vector{Int}
    left_bounds::Vector{Int}
    right_bounds::Vector{Int}
    height::Vector{Float64}
    areas::Vector{Float64}
    nbr_ptr::Vector{Int}
    nbr_idx::Vector{Int}
end

function prefix_sum(v::Vector{Float64})
    n = length(v)
    ps = Vector{Float64}(undef, n + 1)
    ps[1] = 0.0
    @inbounds for i in 1:n
        ps[i + 1] = ps[i] + v[i]
    end
    ps
end

basin_area_ps(ps::Vector{Float64}, left::Int, right::Int, base::Float64) =
    (ps[right+1] - ps[left]) - (right - left + 1) * base

function build_wsdata(minima_indices::Vector{Int}, top_basin_bounds::Dict{Int, Tuple{Int, Int, Float64}}, edges::Set{Tuple{Int, Int}}, signal::Vector{Float64})
    n = length(minima_indices)
    ps = prefix_sum(signal)

    node_ids = copy(minima_indices)
    left_bounds = Vector{Int}(undef, n)
    right_bounds = Vector{Int}(undef, n)
    height = Vector{Float64}(undef, n)
    areas = Vector{Float64}(undef, n)
    
    for (p, id) in enumerate(node_ids)
        l, r, b = top_basin_bounds[id]
        left_bounds[p] = l
        right_bounds[p] = r
        height[p] = b
        areas[p] = basin_area_ps(ps, l, r, b)
    end
    
    id_to_pos = Dict(node_ids[i] => i for i in 1:n)
    
    adj = [Int[] for _ in 1:n]
    for (a, b) in edges
        haskey(id_to_pos, a) && haskey(id_to_pos, b) && push!(adj[id_to_pos[a]], id_to_pos[b])
    end

    nbr_ptr = zeros(Int, n+1)
    for i in 1:n
        nbr_ptr[i+1] = nbr_ptr[i] + length(adj[i])
    end
    nbr_idx = Int[]
    append!(nbr_idx, vcat(adj...))

    WSData(node_ids, left_bounds, right_bounds, height, areas, nbr_ptr, nbr_idx)
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

# -------------------- Plotting --------------------
function plot_level_rectangle!(plt, left::Int, right::Int, bottom::Float64, top::Float64; color=:yellow, opacity=0.25)
    plot!(plt, [left, right, right, left, left], [bottom, bottom, top, top, bottom],
          seriestype=:shape, opacity=opacity, color=color, label=false)
end

function plot_node_rectangle!(plt, ws::WSData, pos::Int, bottom::Float64; color=:yellow, opacity=0.25)
    plot_level_rectangle!(plt, ws.left_bounds[pos], ws.right_bounds[pos], bottom, ws.height[pos]; color=color, opacity=opacity)
end

function plot_node_rectangles!(plt, ws::WSData, positions::Vector{Int}, bottoms::Vector{Float64}; color=:yellow, opacity=0.25)
    @inbounds for (pos, b) in zip(positions, bottoms)
        plot_node_rectangle!(plt, ws, pos, b; color=color, opacity=opacity)
    end
end

function draw_rays!(plt, minima_indices::Vector{Int}, top_basin_bounds::Dict{Int, Tuple{Int, Int, Float64}}, signal::Vector{Float64}, graph_nodes::Dict{Int, GraphNode})
    @inbounds for min_idx in minima_indices
        l, r, val = top_basin_bounds[min_idx]
        plot!(plt, [l, min_idx, r], [val, val, val], linestyle=:dot, color=:purple, lw=1, label=false)
        
        node = graph_nodes[min_idx]
        if node.parent === nothing
            vol_text = string(round(node.area, digits=1))
        else
            vol_text = string(round(node.box_area, digits=1), "+", round(node.side_area, digits=1))
        end
        annotate!(plt, min_idx, signal[min_idx] + 0.05, text(vol_text, 6, :black))
    end
end

function collect_edges(node::GraphNode, edges::Set{Tuple{Int, Int}}=Set{Tuple{Int, Int}}())
    for child in node.children
        push!(edges, (node.id, child.id))
        collect_edges(child, edges)
    end
    return edges
end

function draw_graph_edges!(plt, root::GraphNode, signal::Vector{Float64})
    edges = collect_edges(root)
    for (idx1, idx2) in edges
        plot!(plt, [idx1, idx2], [signal[idx1], signal[idx2]], 
              linestyle=:dash, color=:red, linewidth=2, alpha=0.7, label=false)
    end
end

function draw_node_rectangles!(plt, ws::WSData, positions::Vector{Int}, bottoms::Vector{Float64})
    @inbounds for (pos, b) in zip(positions, bottoms)
        plot_node_rectangle!(plt, ws, pos, b; color=:yellow, opacity=0.25)
    end
end

function draw_ws_rectangles!(plt, ws::WSData)
    if length(ws.node_ids) ≥ 2
        order = sortperm(ws.height)
        pbot = order[1]
        limit = min(length(order), PLOT_MAX_RECTANGLES + 1)
        to_plot = order[2:limit]
        bottoms = overlapping_bottoms(ws, to_plot, pbot)
        plot_node_rectangles!(plt, ws, to_plot, bottoms; color=:yellow, opacity=0.25)
    end
end

# -------------------- Test --------------------
function test_simple()
    signal = [0.8, 0.6, 0.3, 0.15, 0.25, 0.5, 0.7, 0.9, 0.75, 0.4, 0.2, 0.1, 0.18, 0.35, 0.6, 0.8, 0.65, 0.45, 0.25, 0.15]
    
    parent_node = GraphNode(12, 3, 18, 3, 18, 0.1, 0.0, 0.0, 0.0, 0.0, nothing, GraphNode[])
    child_node = GraphNode(4, 3, 18, 7, 16, 0.15, 0.0, 0.0, 0.0, 0.0, parent_node, GraphNode[])
    
    side_area = get_side_area(child_node, signal)
    
    plt = plot(xlabel="Index", ylabel="Height", legend=:topright, grid=true)
    
    plot!(plt, 1:length(signal), signal, lw=3, color=:black, label="Signal", marker=:circle, markersize=4)
    
    hline!(plt, [child_node.height], color=:green, linestyle=:dash, lw=2, label="Node baseline ($(child_node.height))")
    hline!(plt, [parent_node.height], color=:blue, linestyle=:dashdot, lw=2, label="Parent baseline ($(parent_node.height))")
    
    println("\n" * "="^70)
    println("SIDE AREA CALCULATION TEST (using get_side_area function)")
    println("="^70)
    println("Node height: $(child_node.height)")
    println("Parent height: $(parent_node.height)")
    println("Top bounds: [$(child_node.left_i_top), $(child_node.right_i_top)]")
    println("Bottom bounds: [$(child_node.left_i_bottom), $(child_node.right_i_bottom)]")
    
    println("\n" * "-"^70)
    println("LEFT SIDE (indices $(child_node.left_i_top):$(child_node.left_i_bottom-1)):")
    println("-"^70)
    left_area = 0.0
    if child_node.left_i_top < child_node.left_i_bottom
        x_coords = [child_node.left_i_top]
        y_coords = [child_node.height]
        
        for i in child_node.left_i_top:(child_node.left_i_bottom-1)
            clamped_height = max(min(signal[i], child_node.height), parent_node.height)
            contrib = clamped_height - parent_node.height
            left_area += contrib
            println("  i=$i: signal=$(signal[i]), clamped=$(round(clamped_height,digits=3)), contrib=$(round(contrib, digits=3))")
            push!(x_coords, i)
            push!(y_coords, clamped_height)
        end
        
        push!(x_coords, child_node.left_i_bottom-1)
        push!(y_coords, child_node.height)
        
        plot!(plt, x_coords, y_coords, fillrange=child_node.height, fillalpha=0.4, 
              color=:purple, lw=2, label="Left side")
    else
        println("  (empty)")
    end
    println("LEFT TOTAL: $(round(left_area, digits=3))")
    
    println("\n" * "-"^70)
    println("RIGHT SIDE (indices $(child_node.right_i_bottom+1):$(child_node.right_i_top)):")
    println("-"^70)
    right_area = 0.0
    if child_node.right_i_bottom < child_node.right_i_top
        x_coords = [child_node.right_i_bottom+1]
        y_coords = [child_node.height]
        
        for i in (child_node.right_i_bottom+1):child_node.right_i_top
            clamped_height = max(min(signal[i], child_node.height), parent_node.height)
            contrib = clamped_height - parent_node.height
            right_area += contrib
            println("  i=$i: signal=$(signal[i]), clamped=$(round(clamped_height,digits=3)), contrib=$(round(contrib, digits=3))")
            push!(x_coords, i)
            push!(y_coords, clamped_height)
        end
        
        push!(x_coords, child_node.right_i_top)
        push!(y_coords, child_node.height)
        
        plot!(plt, x_coords, y_coords, fillrange=child_node.height, fillalpha=0.4,
              color=:orange, lw=2, label="Right side")
    else
        println("  (empty)")
    end
    println("RIGHT TOTAL: $(round(right_area, digits=3))")
    
    manual_total = left_area + right_area
    
    println("\n" * "="^70)
    println("MANUAL CALCULATION: $(round(manual_total, digits=3))")
    println("get_side_area() RESULT: $(round(side_area, digits=3))")
    println("MATCH: $(isapprox(manual_total, side_area))")
    println("="^70 * "\n")
    
    plot!(plt, size=(1400, 700))
    display(plt)
end

# -------------------- Main --------------------
function main()
    raw_signal = randn(N_POINTS)
    processed_signal = process_signal(raw_signal)

    minima_indices = find_minima_indices(processed_signal)
    maxima_indices = find_maxima_indices(processed_signal)
    top_basin_bounds = get_top_basin_bounds(processed_signal, minima_indices)
    
    root, graph_nodes = build_graph_tree(minima_indices, top_basin_bounds, processed_signal)
    set_graph_areas!(root, processed_signal)
    
    graph_edges = collect_edges(root)
    ws = build_wsdata(minima_indices, top_basin_bounds, graph_edges, processed_signal)

    plt = plot(processed_signal, lw=2, label="Processed Signal", xlabel="Index", ylabel="Amplitude")
    hline!(plt, [0], linestyle=:dash, color=:black, label=false)
    scatter!(plt, minima_indices, processed_signal[minima_indices], color=:green, markersize=2, label="Minima")
    scatter!(plt, maxima_indices, processed_signal[maxima_indices], color=:blue, markersize=2, label="Maxima")
    draw_rays!(plt, minima_indices, top_basin_bounds, processed_signal, graph_nodes)
    draw_graph_edges!(plt, root, processed_signal)
    draw_ws_rectangles!(plt, ws)
    plot(plt, size=(1500, 1000))
    display(plt)    
    
    return nothing
end

main()
