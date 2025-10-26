using Mmap
using Statistics
using InteractiveUtils
using BenchmarkTools
using StaticArrays

function voxel_covariances(data::Array{T,3}) where T
    n_channels, n_voxels, n_subjects = size(data)
    covariances = Array{T}(undef, n_voxels, n_channels, n_channels)
    @views for v in 1:n_voxels
        covariances[v, :, :] = cov(data[:, v, :], dims=2)
    end
    return covariances
end

function voxel_covariances_op(data::Array{T,3}) where T
    n_channels, n_voxels, n_subjects = size(data)
    covariances = Array{T}(undef, n_voxels, n_channels, n_channels)
    mean_vec = Vector{T}(undef, n_channels)
    cov_mat = Matrix{T}(undef, n_channels, n_channels)
    for v in 1:n_voxels
        sum!(mean_vec, @views data[:, v, :])
        mean_vec ./= n_subjects
        fill!(cov_mat, zero(T))
        for i in 1:n_subjects
            @inbounds for ch in 1:n_channels
                diff = data[ch, v, i] - mean_vec[ch]
                for col_channel in 1:n_channels
                    cov_mat[ch, col_channel] += diff * (data[col_channel, v, i] - mean_vec[col_channel])
                end
            end
        end
        cov_mat ./= (n_subjects - 1)
        @views covariances[v, :, :] = cov_mat
    end
    return covariances
end

function arr_mean!(mean_vec::Vector{T}, data::Array{T,3}, v::Int, n_subjects::Int) where T
    sum!(mean_vec, @views data[:, v, :])
    mean_vec ./= n_subjects
end

function arr_mean!(mean_vec::MVector{N,T}, data::Array{T,3}, v::Int, n_subjects::Int) where {T, N}
    sum!(mean_vec, @views data[:, v, :])
    mean_vec ./= n_subjects
end

function update_covariance!(cov_mat::Matrix{T}, data::Array{T,3}, mean_vec::Vector{T}, v::Int, n_subjects::Int, n_channels::Int, zero_mat::Matrix{T}) where T
    cov_mat .= zero_mat
    @inbounds for i in 1:n_subjects
        for row_channel in 1:n_channels
            diff1 = data[row_channel, v, i] - mean_vec[row_channel]
            for col_channel in 1:n_channels
                diff2 = data[col_channel, v, i] - mean_vec[col_channel]
                cov_mat[row_channel, col_channel] += diff1 * diff2
            end
        end
    end
    cov_mat ./= (n_subjects - 1)
end

function update_covariance!(cov_mat::MMatrix{N,N,T}, data::Array{T,3}, mean_vec::MVector{N,T}, v::Int, n_subjects::Int) where {T, N}
    cov_mat .= zero(T)
    @inbounds for i in 1:n_subjects
        for row_channel in 1:N
            diff1 = data[row_channel, v, i] - mean_vec[row_channel]
            for col_channel in 1:N
                diff2 = data[col_channel, v, i] - mean_vec[col_channel]
                cov_mat[row_channel, col_channel] += diff1 * diff2
            end
        end
    end
    cov_mat ./= (n_subjects - 1)
end

function voxel_covariances_op2(data::Array{T,3}) where T
    n_channels, n_voxels, n_subjects = size(data)
    covariances = Array{T}(undef, n_voxels, n_channels, n_channels)
    mean_vec = Vector{T}(undef, n_channels)
    cov_mat = Matrix{T}(undef, n_channels, n_channels)
    zero_mat = zeros(T, n_channels, n_channels)

    for v in 1:n_voxels
        arr_mean!(mean_vec, data, v, n_subjects)
        update_covariance!(cov_mat, data, mean_vec, v, n_subjects, n_channels, zero_mat)
        @views covariances[v, :, :] = cov_mat
    end

    return covariances
end


function voxel_covariances_static(data::Array{T,3}) where T
    n_channels, n_voxels, n_subjects = size(data)
    covariances = Array{T}(undef, n_voxels, n_channels, n_channels)
    
    mean_vec = @MVector zeros(T, 3)
    cov_mat = @MMatrix zeros(T, 3, 3)
    
    for v in 1:n_voxels
        arr_mean!(mean_vec, data, v, n_subjects)
        update_covariance!(cov_mat, data, mean_vec, v, n_subjects)
        @views covariances[v, :, :] = cov_mat
    end


    return covariances
end


x = rand(Float32, 3, 1_000, 100);

@assert all(isapprox.(voxel_covariances(x), voxel_covariances_op(x), rtol=1e-5, atol=1e-6))
@assert all(isapprox.(voxel_covariances(x), voxel_covariances_op2(x), rtol=1e-5, atol=1e-6))
@assert all(isapprox.(voxel_covariances(x), voxel_covariances_static(x), rtol=1e-5, atol=1e-6))
@time voxel_covariances(x);
@time voxel_covariances_op(x);
@time voxel_covariances_op2(x);
@time voxel_covariances_static(x);
nothing

# @code_warntype voxel_means(x)