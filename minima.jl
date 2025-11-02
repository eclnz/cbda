include("coord.jl")
using BenchmarkTools

function find_local_minima_along_axes!(
    output::Vector{Coord1D{Int}},
    data::Vector{T};
    exclude_border::Bool=true
) where {T}
    len = length(data)
    start_idx = exclude_border ? 2 : 1
    end_idx   = exclude_border ? len - 1 : len

    count = 0
    @inbounds @simd for i in start_idx:end_idx
        val = data[i]
        is_min = (i == 1 || data[i-1] > val) && (i == len || data[i+1] > val)
        is_max = (i == 1 || data[i-1] < val) && (i == len || data[i+1] < val)
        if is_min || is_max
            count += 1
            output[count] = Coord1D(i)
        end
    end
    return count
end


function find_local_minima_along_axes!(
    output::Vector{Coord2D{Int}},
    data::AbstractMatrix{T};
    exclude_border::Bool=true
) where {T}
    nx, ny = size(data)
    i_start = exclude_border ? 2 : 1
    i_end   = exclude_border ? nx - 1 : nx
    j_start = exclude_border ? 2 : 1
    j_end   = exclude_border ? ny - 1 : ny

    count = 0
    @inbounds for j in j_start:j_end
        @simd for i in i_start:i_end
            val = data[i, j]
            is_min = data[i-1, j] > val &&
                     data[i+1, j] > val &&
                     data[i, j-1] > val &&
                     data[i, j+1] > val
            is_max = data[i-1, j] < val &&
                     data[i+1, j] < val &&
                     data[i, j-1] < val &&
                     data[i, j+1] < val
            if is_min || is_max
                count += 1
                output[count] = Coord2D(i, j)
            end
        end
    end
    return count
end


function find_local_minima_along_axes!(
    output::Vector{Coord3D{Int}},
    data::AbstractArray{T,3};
    exclude_border::Bool=true
) where {T}
    nx, ny, nz = size(data)
    i_start = exclude_border ? 2 : 1
    i_end   = exclude_border ? nx - 1 : nx
    j_start = exclude_border ? 2 : 1
    j_end   = exclude_border ? ny - 1 : ny
    k_start = exclude_border ? 2 : 1
    k_end   = exclude_border ? nz - 1 : nz

    count = 0
    @inbounds for k in k_start:k_end
        for j in j_start:j_end
            @simd for i in i_start:i_end
                val = data[i, j, k]
                is_min = data[i-1, j, k] > val &&
                         data[i+1, j, k] > val &&
                         data[i, j-1, k] > val &&
                         data[i, j+1, k] > val &&
                         data[i, j, k-1] > val &&
                         data[i, j, k+1] > val
                is_max = data[i-1, j, k] < val &&
                         data[i+1, j, k] < val &&
                         data[i, j-1, k] < val &&
                         data[i, j+1, k] < val &&
                         data[i, j, k-1] < val &&
                         data[i, j, k+1] < val
                if is_min || is_max
                    count += 1
                    output[count] = Coord3D(i, j, k)
                end
            end
        end
    end
    return count
end


function find_local_minima_along_axes(data::Vector{T}; exclude_border::Bool=true) where {T}
    output = Vector{Coord1D{Int}}(undef, length(data))
    count = find_local_minima_along_axes!(output, data; exclude_border=exclude_border)
    return output[1:count]
end

function find_local_minima_along_axes(data::AbstractMatrix{T}; exclude_border::Bool=true) where {T}
    output = Vector{Coord2D{Int}}(undef, length(data))
    count = find_local_minima_along_axes!(output, data; exclude_border=exclude_border)
    return output[1:count]
end

function find_local_minima_along_axes(data::AbstractArray{T,3}; exclude_border::Bool=true) where {T}
    output = Vector{Coord3D{Int}}(undef, length(data))
    count = find_local_minima_along_axes!(output, data; exclude_border=exclude_border)
    return output[1:count]
end


# minima = find_local_minima_along_axes(rand(100))
# minima = find_local_minima_along_axes(rand(100, 100))
# minima = find_local_minima_along_axes(rand(100, 100, 100))

# @btime minima = find_local_minima_along_axes(rand(100))
# @btime minima = find_local_minima_along_axes(rand(100, 100))
# @btime minima = find_local_minima_along_axes(rand(100, 100, 100))

nothing
