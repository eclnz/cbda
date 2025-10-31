using Random
using Plots

include("conv.jl")

function visualize_sides(; max_nodes::Int=5)
    Random.seed!(1234)
    npts = 160
    raw = randn(npts)
    sig = process_signal(raw)

    mins = find_minima_indices(sig)
    top_bounds = get_top_basin_bounds(sig, mins)
    root, nodes = build_graph_tree(mins, top_bounds, sig)
    set_graph_areas!(root, sig)

    cands = [n for (_, n) in nodes if n.parent !== nothing]
    isempty(cands) && error("No child nodes found to visualize")
    order = sortperm(map(n -> n.side_area, cands); rev=true)
    pick = cands[order[1: min(max_nodes, length(order))]]

    plt = plot(sig, lw=2, label="Signal", xlabel="Index", ylabel="Amplitude")

    # Use library side-area helpers
    side_area(side::Symbol) = side == :left ? get_left_area(node, sig) : get_right_area(node, sig)

    function fill_side!(node::GraphNode, side::Symbol; add_labels::Bool=false)
        parent = node.parent
        n = length(sig)
        if side == :left
            lstart = node.left_i_top + 1
            lend = node.left_i_bottom - 1
            lstart = clamp(lstart, 1, n)
            lend = clamp(lend, 0, n)
            if lstart > lend
                return
            end
            xs = Int[]; ys = Float64[]
            push!(xs, lstart); push!(ys, parent.height)
            for i in lstart:lend
                clamped = max(min(sig[i], node.height), parent.height)
                push!(xs, i); push!(ys, clamped)
            end
            push!(xs, lend); push!(ys, parent.height)
            plot!(plt, xs, ys, seriestype=:shape, alpha=0.35, color=:purple, label=add_labels ? "Left side" : false)
        else
            rstart = node.right_i_top + 1
            rend = node.right_i_bottom - 1
            rstart = clamp(rstart, 1, n)
            rend = clamp(rend, 0, n)
            if rstart > rend
                return
            end
            xs = Int[]; ys = Float64[]
            push!(xs, rstart); push!(ys, parent.height)
            for i in rstart:rend
                clamped = max(min(sig[i], node.height), parent.height)
                push!(xs, i); push!(ys, clamped)
            end
            push!(xs, rend); push!(ys, parent.height)
            plot!(plt, xs, ys, seriestype=:shape, alpha=0.35, color=:orange, label=add_labels ? "Right side" : false)
        end
    end

    first = true
    for node in pick
        parent = node.parent
        hline!(plt, [parent.height], color=:blue, linestyle=:dash, label=first ? "Parent height" : false)
        hline!(plt, [node.height], color=:green, linestyle=:dash, label=first ? "Node height" : false)
        vline!(plt, [node.left_i_top, node.right_i_top], color=:purple, linestyle=:dot, label=false)
        vline!(plt, [node.left_i_bottom, node.right_i_bottom], color=:orange, linestyle=:dashdot, label=false)
        fill_side!(node, :left; add_labels=first)
        fill_side!(node, :right; add_labels=first)
        plot_level_rectangle!(plt, node.left_i_top, node.right_i_top, parent.height, node.height; color=:yellow, opacity=0.2)
        l_area = get_left_area(node, sig)
        r_area = get_right_area(node, sig)
        annotate!(plt, node.id, sig[node.id] + 0.05, text("$(node.id): box=$(round(node.box_area, digits=2)) L=$(round(l_area, digits=2)) R=$(round(r_area, digits=2))", 7, :black))
        println("Node $(node.id) parent $(parent.id) top=[$(node.left_i_top), $(node.right_i_top)] bottom=[$(node.left_i_bottom), $(node.right_i_bottom)] box=$(node.box_area) left=$(l_area) right=$(r_area) side=$(node.side_area) area=$(node.area)")
        first = false
    end

    plot!(plt, size=(1800, 1100))
    display(plt)
end

visualize_sides()


