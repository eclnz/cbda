using Random
using Plots
using DataStructures

const SEED = 123
const N_POINTS = 100
const KERNEL_SIZE = 21
const KERNEL_SIGMA = 3.0
const ADD_OFFSET = true
const OFFSET_VALUE = 0.1
const POINT_OF_INTEREST = 77
const DISCRETE_LEVELS = 50

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
    divisions = Vector{Tuple{Int, Int, Float64}}()
    for min_idx in minima_indices
        target_val = signal[min_idx]
        left = project_ray_left(signal, min_idx, target_val)
        right = project_ray_right(signal, min_idx, target_val)
        push!(divisions, (left, right, target_val))
    end
    return divisions
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

divisions = project_rays_from_minima(smooth_signal, minima_indices)

p1 = plot(smooth_signal, label="Smooth Signal", xlabel="Index", ylabel="Amplitude", lw=2)
hline!(p1, [0], linestyle=:dash, color=:black, label="Zero")
if 1 <= POINT_OF_INTEREST <= length(smooth_signal)
    vline!(p1, [POINT_OF_INTEREST], linestyle=:dash, color=:red, label="POI", lw=2)
end
if length(minima_indices) > 0
    scatter!(p1, minima_indices, smooth_signal[minima_indices], color=:green, markersize=8, markershape=:x, label="Local Minima")
end
if length(maxima_indices) > 0
    scatter!(p1, maxima_indices, smooth_signal[maxima_indices], color=:blue, markersize=8, markershape=:x, label="Local Maxima")
end

for (min_idx, (left, right, target_val)) in zip(minima_indices, divisions)
    plot!(p1, [left, min_idx, right], [target_val, target_val, target_val], 
          linestyle=:dot, color=:purple, linewidth=1, label="")
end

plot(p1, layout=(2,1), size=(1000,1000))
