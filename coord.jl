struct Coord1D{T}
    x::T
end

struct Coord2D{T}
    x::T
    y::T
end

struct Coord3D{T}
    x::T
    y::T
    z::T
end

unit_directions(::Type{Coord1D}) = (-1, 1)
unit_directions(::Type{Coord2D}) = (Coord2D(dx, dy) for dx in -1:1, dy in -1:1 if !(dx == 0 && dy == 0))
unit_directions(::Type{Coord3D}) = (Coord3D(dx, dy, dz) for dx in -1:1, dy in -1:1, dz in -1:1 if !(dx == 0 && dy == 0 && dz == 0))

function project_ray(data::Vector{T}, coord::Coord1D{<:Integer}, direction::Int; epsval=eps(Float64)) where T
    i = coord.x + direction
    while 1 ≤ i ≤ length(data)
        if data[i] < 0 || abs(data[i]) < epsval
            return Coord1D(i)
        end
        i += direction
    end
    return Coord1D(direction == -1 ? 1 : length(data))
end

function project_ray(data::AbstractArray{T,2}, coord::Coord2D{<:Integer}, direction::Coord2D{<:Integer}; epsval=eps(Float64)) where T
    i = coord.x + direction.x
    j = coord.y + direction.y
    while 1 ≤ i ≤ size(data, 1) && 1 ≤ j ≤ size(data, 2)
        if data[i, j] < 0 || abs(data[i, j]) < epsval
            return Coord2D(i, j)
        end
        i += direction.x
        j += direction.y
    end
    return Coord2D(i - direction.x, j - direction.y)
end

function project_ray(data::AbstractArray{T,3}, coord::Coord3D{<:Integer}, direction::Coord3D{<:Integer}; epsval=eps(Float64)) where T
    i = coord.x + direction.x
    j = coord.y + direction.y
    k = coord.z + direction.z
    while 1 ≤ i ≤ size(data, 1) && 1 ≤ j ≤ size(data, 2) && 1 ≤ k ≤ size(data, 3)
        if data[i, j, k] < 0 || abs(data[i, j, k]) < epsval
            return Coord3D(i, j, k)
        end
        i += direction.x
        j += direction.y
        k += direction.z
    end
    return Coord3D(i - direction.x, j - direction.y, k - direction.z)
end

function project_all_rays(data, coord::C; epsval=eps(Float64)) where {C<:Union{Coord1D{<:Integer}, Coord2D{<:Integer}, Coord3D{<:Integer}}}
    directions = collect(unit_directions(C.name.wrapper))
    coords = Vector{C}(undef, length(directions))
    for (idx, direction) in enumerate(directions)
        coords[idx] = project_ray(data, coord, direction; epsval=epsval)
    end
    return coords
end
