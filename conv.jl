using Random
using Plots
using DataStructures

const SEED = 124
const N_POINTS = 120
const KERNEL_SIZE = 21
const KERNEL_SIGMA = 2.5
const ADD_OFFSET = true
const OFFSET_VALUE = 0.1
const DISCRETE_LEVELS = 200

Random.seed!(SEED)
raw_signal::Vector{Float64} = randn(N_POINTS)

function conv_valid(signal::Vector{Float64}, kernel::Vector{Float64})::Vector{Float64}
    pad = div(length(kernel), 2)
    padded_signal = vcat(fill(signal[1], pad), signal, fill(signal[end], pad))
    n = length(padded_signal) - length(kernel) + 1
    result = Vector{Float64}(undef, n)
    for i in 1:n
        result[i] = sum(padded_signal[i:i+length(kernel)-1] .* kernel)
    end
    return result
end

function gaussian_kernel(size::Int, sigma::Float64)::Vector{Float64}
    x = range(-div(size, 2), stop=div(size, 2), length=size)
    kernel = exp.(-x.^2 ./ (2*sigma^2))
    return kernel ./ sum(kernel)
end

function discretize_amplitude(signal::Vector{Float64}, levels::Int)::Vector{Float64}
    min_val = minimum(signal)
    max_val = maximum(signal)
    range_val = max_val - min_val
    
    discrete = Vector{Float64}(undef, length(signal))
    for i in 1:length(signal)
        normalized = (signal[i] - min_val) / range_val
        discrete[i] = floor(normalized * levels) / levels * range_val + min_val
    end
    
    return discrete
end

function compute_gradient(signal::Vector{Float64})::Vector{Float64}
    n = length(signal)
    gradient = Vector{Float64}(undef, n)
    gradient[1] = signal[2] - signal[1]
    for i in 2:(n-1)
        gradient[i] = (signal[i+1] - signal[i-1]) / 2.0
    end
    gradient[n] = signal[n] - signal[n-1]
    return gradient
end

function find_min_index(signal::Vector{Float64}, i::Int, j::Int)::Int
    signal[i] < signal[j] ? i : j
end

function find_max_index(signal::Vector{Float64}, i::Int, j::Int)::Int
    signal[i] > signal[j] ? i : j
end

function find_extrema(signal::Vector{Float64}, gradient::Vector{Float64})
    maxima_pq = PriorityQueue{Int, Float64}()
    minima_pq = PriorityQueue{Int, Float64}()
    
    for i in 1:(length(gradient)-1)
        if gradient[i] <= 0 && gradient[i+1] > 0
            idx = find_min_index(signal, i, i+1)
            enqueue!(minima_pq, idx, signal[idx])
        elseif gradient[i] >= 0 && gradient[i+1] < 0
            idx = find_max_index(signal, i, i+1)
            enqueue!(maxima_pq, idx, -signal[idx])
        end
    end
    
    return minima_pq, maxima_pq
end

function project_ray_left(signal::Vector{Float64}, idx::Int, target_val::Float64)
    for i in (idx-1):-1:1
        if signal[i] < target_val
            return i
        elseif signal[i] > target_val
            for j in i:-1:1
                if signal[j] <= target_val
                    return j
                end
            end
            return 1
        end
    end
    return 1
end

function project_ray_right(signal::Vector{Float64}, idx::Int, target_val::Float64)
    for i in (idx+1):length(signal)
        if signal[i] < target_val
            return i
        elseif signal[i] > target_val
            for j in i:length(signal)
                if signal[j] <= target_val
                    return j
                end
            end
            return length(signal)
        end
    end
    return length(signal)
end

function project_rays_from_minima(signal::Vector{Float64}, minima_indices::Vector{Int})
    minima_coords = Dict{Int, Tuple{Int, Int, Float64}}()
    for min_idx in minima_indices
        target_val = signal[min_idx]
        left = project_ray_left(signal, min_idx, target_val)
        right = project_ray_right(signal, min_idx, target_val)
        minima_coords[min_idx] = (left, right, target_val)
    end
    return minima_coords
end

function find_neighbors_at_level(min_idx::Int, candidate_indices::Vector{Int}, signal::Vector{Float64}, minima_coords::Dict{Int, Tuple{Int, Int, Float64}})
    left_bound, right_bound, _ = minima_coords[min_idx]
    left_neighbor = nothing
    right_neighbor = nothing
    
    for idx in candidate_indices
        idx == min_idx && continue
        
        if idx >= left_bound && idx <= right_bound
            if idx < min_idx && (left_neighbor === nothing || signal[idx] < signal[left_neighbor])
                left_neighbor = idx
            elseif idx > min_idx && (right_neighbor === nothing || signal[idx] < signal[right_neighbor])
                right_neighbor = idx
            end
        end
    end
    
    return left_neighbor, right_neighbor
end

function add_edge_pair(edges::Set{Tuple{Int, Int}}, idx1::Int, idx2::Int)
    push!(edges, (idx1, idx2))
    push!(edges, (idx2, idx1))
end

function build_watershed_graph(minima_indices::Vector{Int}, signal::Vector{Float64}, minima_coords::Dict{Int, Tuple{Int, Int, Float64}}, max_iter::Int=100)
    edges = Set{Tuple{Int, Int}}()
    edge_info = Vector{Tuple{Int, Int, String}}()
    
    isempty(minima_indices) && return edges, edge_info

    sorted_minima = sort(minima_indices, by=x->signal[x])
    processed = Set{Int}()
    
    for (iter, min_idx) in enumerate(sorted_minima[1:min(length(sorted_minima), max_iter)])
        candidates = [idx for idx in sorted_minima if !(idx in processed)]
        left_neighbor, right_neighbor = find_neighbors_at_level(min_idx, candidates, signal, minima_coords)
        
        if left_neighbor !== nothing
            add_edge_pair(edges, min_idx, left_neighbor)
            push!(edge_info, (min_idx, left_neighbor, "iter_$iter"))
        end
        
        if right_neighbor !== nothing
            add_edge_pair(edges, min_idx, right_neighbor)
            push!(edge_info, (min_idx, right_neighbor, "iter_$iter"))
        end
        
        push!(processed, min_idx)
        if left_neighbor !== nothing
            push!(processed, left_neighbor)
        end
        if right_neighbor !== nothing
            push!(processed, right_neighbor)
        end
    end

    return edges, edge_info
end

kernel::Vector{Float64} = gaussian_kernel(KERNEL_SIZE, KERNEL_SIGMA)
smooth_signal::Vector{Float64} = conv_valid(raw_signal, kernel)
if ADD_OFFSET
    smooth_signal .+= abs(minimum(smooth_signal)) + OFFSET_VALUE
end
smooth_signal = discretize_amplitude(smooth_signal, DISCRETE_LEVELS)

gradient_signal::Vector{Float64} = compute_gradient(smooth_signal)

minima_pq, maxima_pq = find_extrema(smooth_signal, gradient_signal)

minima_indices = collect(keys(minima_pq))
maxima_indices = collect(keys(maxima_pq))

minima_coords = project_rays_from_minima(smooth_signal, minima_indices)

graph_edges, edge_info = build_watershed_graph(minima_indices, smooth_signal, minima_coords)

p1 = plot(smooth_signal, label="Smooth Signal", xlabel="Index", ylabel="Amplitude", lw=2)
hline!(p1, [0], linestyle=:dash, color=:black, label="Zero")
if length(minima_indices) > 0
    scatter!(p1, minima_indices, smooth_signal[minima_indices], color=:green, markersize=8, markershape=:x, label="Local Minima")
end
if length(maxima_indices) > 0
    scatter!(p1, maxima_indices, smooth_signal[maxima_indices], color=:blue, markersize=8, markershape=:x, label="Local Maxima")
end

left_boundaries = Int[]
right_boundaries = Int[]

for min_idx in minima_indices
    left, right, target_val = minima_coords[min_idx]
    plot!(p1, [left, min_idx, right], [target_val, target_val, target_val], 
          linestyle=:dot, color=:purple, linewidth=1, label="")
    push!(left_boundaries, left)
    push!(right_boundaries, right)
end

scatter!(p1, left_boundaries, smooth_signal[left_boundaries], color=:cyan, markersize=4, markershape=:circle, label="Left Bounds")
scatter!(p1, right_boundaries, smooth_signal[right_boundaries], color=:magenta, markersize=4, markershape=:circle, label="Right Bounds")

colors = Dict("iter_1" => :red, "iter_2" => :orange, "iter_3" => :yellow)

for (idx1, idx2, iter_str) in edge_info
    color = haskey(colors, iter_str) ? colors[iter_str] : :orange
    plot!(p1, [idx1, idx2], [smooth_signal[idx1], smooth_signal[idx2]], 
          linestyle=:dash, color=color, linewidth=1, alpha=0.7, label="")
end

plot(p1, layout=(2,1), size=(1000,1000))
