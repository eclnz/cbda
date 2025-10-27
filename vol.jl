function voxel_stats_to_volume(voxel_stats::Array{T,1}, x::Int, y::Int, z::Int) where T
    n_voxels = length(voxel_stats)
    @assert n_voxels == x * y * z "n_voxels must equal x * y * z"
    
    volume = Array{T}(undef, x, y, z)
    
    voxel_idx = 1
    for zi in 1:z, yi in 1:y, xi in 1:x
        volume[xi, yi, zi] = voxel_stats[voxel_idx]
        voxel_idx += 1
    end
    
    return volume
end

function volume_to_voxel_stats(volume::Array{T,3}) where T
    x, y, z = size(volume)
    n_voxels = x * y * z
    
    voxel_stats = Array{T}(undef, n_voxels)
    
    voxel_idx = 1
    for zi in 1:z, yi in 1:y, xi in 1:x
        voxel_stats[voxel_idx] = volume[xi, yi, zi]
        voxel_idx += 1
    end
    
    return voxel_stats
end
