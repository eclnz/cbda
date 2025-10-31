include("conv.jl")

using Test

@testset "Watershed Algorithm Tests" begin
    
    @testset "GraphNode Construction" begin
        node = GraphNode(5, 1, 10, 0.5)
        @test node.id == 5
        @test node.left_i_top == 1
        @test node.right_i_top == 10
        @test node.left_i_bottom == 1
        @test node.right_i_bottom == 10
        @test node.height == 0.5
        @test node.box_area == 0.0
        @test node.side_area == 0.0
        @test node.area == 0.0
        @test node.cumul_area == 0.0
        @test node.parent === nothing
        @test length(node.children) == 0
    end
    
    @testset "Node Width Calculation" begin
        node = GraphNode(5, 3, 10, 0.5)
        @test get_node_width(node) == 8
        
        node2 = GraphNode(1, 1, 1, 0.5)
        @test get_node_width(node2) == 1
    end
    
    @testset "Side Area Calculation" begin
        signal = [0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.5, 0.3, 0.2, 0.4]
        
        parent = GraphNode(1, 1, 10, 1, 10, 0.1, 0.0, 0.0, 0.0, 0.0, nothing, GraphNode[])
        child = GraphNode(3, 1, 10, 1, 5, 0.2, 0.0, 0.0, 0.0, 0.0, parent, GraphNode[])
        
        side_area = get_side_area(child, signal)
        
        left_expected = 0.0
        right_expected = 0.0
        for i in 6:10
            clamped = max(min(signal[i], child.height), parent.height)
            right_expected += clamped - parent.height
        end
        
        @test side_area ≈ right_expected
    end
    
    @testset "Box and Side Area Calculation" begin
        signal = [0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.5, 0.3, 0.2, 0.4]
        
        parent = GraphNode(1, 1, 10, 1, 10, 0.1, 0.0, 0.0, 0.0, 0.0, nothing, GraphNode[])
        child = GraphNode(3, 1, 10, 1, 5, 0.2, 0.0, 0.0, 0.0, 0.0, parent, GraphNode[])
        
        set_graph_areas!(parent, signal)
        set_graph_areas!(child, signal)
        
        @test parent.box_area ≈ 0.1 * 10
        @test parent.side_area == 0.0
        @test parent.area ≈ parent.box_area
        
        expected_box = 10 * (0.2 - 0.1)
        @test child.box_area ≈ expected_box
        @test child.area ≈ child.box_area + child.side_area
    end
    
    @testset "Cumulative Area Calculation" begin
        signal = [0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.5, 0.3, 0.2, 0.4]
        
        parent = GraphNode(1, 1, 10, 1, 10, 0.1, 0.0, 0.0, 0.0, 0.0, nothing, GraphNode[])
        child1 = GraphNode(3, 1, 5, 1, 3, 0.2, 0.0, 0.0, 0.0, 0.0, parent, GraphNode[])
        child2 = GraphNode(7, 6, 10, 8, 10, 0.15, 0.0, 0.0, 0.0, 0.0, parent, GraphNode[])
        
        push!(parent.children, child1)
        push!(parent.children, child2)
        
        set_graph_areas!(parent, signal)
        
        @test parent.cumul_area ≈ parent.area + child1.cumul_area + child2.cumul_area
    end
    
    @testset "Parent-Child Relationship" begin
        signal = [0.5, 0.3, 0.1, 0.2, 0.4]
        
        parent = GraphNode(1, 1, 5, 1, 5, 0.1, 0.0, 0.0, 0.0, 0.0, nothing, GraphNode[])
        child = GraphNode(3, 2, 4, 2, 3, 0.2, 0.0, 0.0, 0.0, 0.0, parent, GraphNode[])
        push!(parent.children, child)
        
        @test child.parent === parent
        @test parent in [child.parent]
        @test child in parent.children
        @test child.height > parent.height
    end
    
    @testset "Side Area Bounds Clamping" begin
        signal = [1.0, 0.8, 0.5, 0.3, 0.6, 0.9, 1.0]
        
        parent = GraphNode(1, 1, 7, 1, 7, 0.2, 0.0, 0.0, 0.0, 0.0, nothing, GraphNode[])
        child = GraphNode(4, 1, 7, 3, 5, 0.5, 0.0, 0.0, 0.0, 0.0, parent, GraphNode[])
        
        side_area = get_side_area(child, signal)
        
        left_manual = 0.0
        for i in 1:2
            clamped = max(min(signal[i], 0.5), 0.2)
            left_manual += clamped - 0.2
        end
        
        right_manual = 0.0
        for i in 6:7
            clamped = max(min(signal[i], 0.5), 0.2)
            right_manual += clamped - 0.2
        end
        
        @test side_area ≈ left_manual + right_manual
    end
    
    @testset "Area Components Sum" begin
        signal = [0.8, 0.6, 0.3, 0.15, 0.25, 0.5, 0.7, 0.9, 0.75, 0.4]
        
        parent = GraphNode(3, 1, 10, 1, 10, 0.1, 0.0, 0.0, 0.0, 0.0, nothing, GraphNode[])
        child = GraphNode(4, 1, 10, 4, 7, 0.2, 0.0, 0.0, 0.0, 0.0, parent, GraphNode[])
        push!(parent.children, child)
        
        set_graph_areas!(parent, signal)
        
        @test child.area ≈ child.box_area + child.side_area atol=1e-10
        @test parent.area ≈ parent.box_area + parent.side_area atol=1e-10
    end
    
    @testset "Graph Nodes Dict Preserves References" begin
        signal = [0.5, 0.3, 0.1, 0.2, 0.4]
        
        nodes = Dict{Int, GraphNode}()
        nodes[3] = GraphNode(3, 1, 5, 0.1)
        
        set_graph_areas!(nodes[3], signal)
        
        @test nodes[3].area > 0.0
        @test nodes[3].box_area > 0.0
    end
    
    @testset "Full Pipeline: Build Tree -> Set Areas -> Check Values" begin
        Random.seed!(102)
        signal = randn(20)
        processed = process_signal(signal)
        
        minima_indices = find_minima_indices(processed)
        basin_bounds = get_top_basin_bounds(processed, minima_indices)
        
        root, graph_nodes = build_graph_tree(minima_indices, basin_bounds, processed)
        
        println("\nBefore set_graph_areas!:")
        for idx in minima_indices[1:min(3, length(minima_indices))]
            node = graph_nodes[idx]
            println("  Node $idx: box=$(node.box_area), side=$(node.side_area), area=$(node.area)")
        end
        
        set_graph_areas!(root, processed)
        
        println("\nAfter set_graph_areas!:")
        for idx in minima_indices[1:min(3, length(minima_indices))]
            node = graph_nodes[idx]
            println("  Node $idx: box=$(node.box_area), side=$(node.side_area), area=$(node.area)")
            
            if node.parent !== nothing
                @test node.area ≈ node.box_area + node.side_area atol=1e-10
                @test node.box_area ≥ 0.0
                @test node.side_area ≥ 0.0
            end
        end
        
        println("\nSimulating draw_rays! display:")
        for min_idx in minima_indices[1:min(3, length(minima_indices))]
            node = graph_nodes[min_idx]
            if node.parent === nothing
                vol_text = string(round(node.area, digits=1))
            else
                vol_text = string(round(node.box_area, digits=1), "+", round(node.side_area, digits=1))
            end
            println("  Node $min_idx would display: $vol_text")
        end
    end

    @testset "Bounds Invariants On Built Tree" begin
        Random.seed!(102)
        signal = process_signal(randn(120))
        minima_indices = find_minima_indices(signal)
        basin_bounds = get_top_basin_bounds(signal, minima_indices)
        root, nodes = build_graph_tree(minima_indices, basin_bounds, signal)
        set_graph_areas!(root, signal)
        for (_, n) in nodes
            @test n.left_i_top <= n.right_i_top
            @test n.left_i_bottom <= n.right_i_bottom
            @test n.left_i_top <= n.left_i_bottom <= n.right_i_bottom <= n.right_i_top
        end
    end

end

println("\n✓ All tests passed!")

@testset "Helper Functions Tests" begin
    include("conv.jl")

    @testset "compute_bottom_bounds left/right cases" begin
        child = GraphNode(40, 10, 50, 0.3)
        parent = GraphNode(60, 1, 120, 0.1)
        lb, rb = compute_bottom_bounds(child, parent)
        @test lb == child.left_i_top == 10
        @test rb == child.left_i_top == 10
        @test lb <= rb

        child2 = GraphNode(80, 30, 90, 0.35)
        parent2 = GraphNode(60, 1, 120, 0.2)
        lb2, rb2 = compute_bottom_bounds(child2, parent2)
        @test lb2 == child2.left_i_top == 30
        @test rb2 == min(child2.right_i_top, parent2.id) == 60
        @test lb2 <= rb2

        # Degenerate case: parent index outside child's top bounds
        child3 = GraphNode(20, 40, 45, 0.25)
        parent3 = GraphNode(10, 1, 120, 0.1)
        lb3, rb3 = compute_bottom_bounds(child3, parent3)
        @test lb3 == child3.left_i_top
        @test rb3 == child3.left_i_top
        @test lb3 <= rb3
    end

    @testset "find_parent containment and height ordering" begin
        signal = [0.5, 0.3, 0.2, 0.4, 0.6, 0.25, 0.35]
        minima = [3, 6]  # indices 3 (0.2), 6 (0.25)
        bounds = Dict{Int,Tuple{Int,Int,Float64}}(
            3 => (2, 7, signal[3]),  # contains child's basin
            6 => (4, 6, signal[6])
        )
        nodes = Dict{Int,GraphNode}()
        for m in minima
            l, r, b = bounds[m]
            nodes[m] = GraphNode(m, l, r, b)
        end
        sorted_minima = sort(minima, by=x->signal[x])
        processed = Set{Int}([sorted_minima[1]])
        cand = find_parent(sorted_minima[2], sorted_minima, processed, signal, bounds, nodes)
        @test cand !== nothing
        @test cand.id == 3

        # If processed empty, no parent
        processed2 = Set{Int}()
        cand2 = find_parent(6, sorted_minima, processed2, signal, bounds, nodes)
        @test cand2 === nothing
    end
end

