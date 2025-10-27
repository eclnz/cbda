using Mmap
using Statistics
using InteractiveUtils
using StaticArrays
using LinearAlgebra
using LoopVectorization
using BenchmarkTools

function voxel_covariances(data::Array{T,3}) where T
    n_channels, n_voxels, n_subjects = size(data)
    covariances = Array{T}(undef, n_voxels)
    @views for v in 1:n_voxels
        covariances[v] = tr(cov(data[:, v, :], dims=2))
    end
    return covariances
end

function voxel_covariances_op(data::Array{T,3}) where T
    n_channels, n_voxels, n_subjects = size(data)
    variances = Array{T}(undef, n_voxels)
    mean_vec = Vector{T}(undef, n_channels)
    cov_mat = Matrix{T}(undef, n_channels, n_channels)
    for voxel_i in 1:n_voxels
        sum!(mean_vec, @views data[:, voxel_i, :])
        mean_vec ./= n_subjects
        fill!(cov_mat, zero(T))
        for i in 1:n_subjects
            @inbounds for ch in 1:n_channels
                diff = data[ch, voxel_i, i] - mean_vec[ch]
                for col_channel in 1:n_channels
                    cov_mat[ch, col_channel] += diff * (data[col_channel, voxel_i, i] - mean_vec[col_channel])
                end
            end
        end
        cov_mat ./= (n_subjects - 1)
        variances[voxel_i] = tr(cov_mat)
    end
    return variances
end

function arr_mean!(mean_vec::Vector{T}, data::Array{T,3}, v::Int, n_subjects::Int) where T
    sum!(mean_vec, @views data[:, v, :])
    mean_vec ./= n_subjects
end

function arr_mean!(mean_vec::MVector{N,T}, data::Array{T,3}, v::Int, n_subjects::Int) where {T, N}
    sum!(mean_vec, @views data[:, v, :])
    mean_vec ./= n_subjects
end

function arr_mean!(mean_vec::MVector{N,T}, data::Array{T,3}, v::Int, n_subjects::Int, subject_indices::AbstractVector{<:Integer}) where {T, N}
    mean_vec .= zero(T)
    n = length(subject_indices)
    @inbounds @avx for i in 1:n
        idx = subject_indices[i]
        for ch in 1:N
            mean_vec[ch] += data[ch, v, idx]
        end
    end
    mean_vec ./= n_subjects
end

function arr_covariance!(cov_mat::Matrix{T}, data::Array{T,3}, mean_vec::Vector{T}, voxel_i::Int, n_subjects::Int, n_channels::Int, zero_mat::Matrix{T}) where T
    cov_mat .= zero_mat
    @inbounds for i in 1:n_subjects
        for row_channel in 1:n_channels
            diff1 = data[row_channel, voxel_i, i] - mean_vec[row_channel]
            for col_channel in 1:n_channels
                diff2 = data[col_channel, voxel_i, i] - mean_vec[col_channel]
                cov_mat[row_channel, col_channel] += diff1 * diff2
            end
        end
    end
    cov_mat ./= (n_subjects - 1)
end

function arr_covariance!(cov_mat::MMatrix{N,N,T}, data::Array{T,3}, mean_vec::MVector{N,T}, v::Int, n_subjects::Int) where {T, N}
    @inbounds @avx for i in 1:n_subjects
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

function arr_covariance!(cov_mat::MMatrix{N,N,T}, data::Array{T,3}, mean_vec::MVector{N,T}, v::Int, n_subjects::Int, subject_indices::AbstractVector{<:Integer}) where {T, N}
    n = length(subject_indices)
    @inbounds @avx for i in 1:n
        idx = subject_indices[i]
        for row_channel in 1:N
            diff1 = data[row_channel, v, idx] - mean_vec[row_channel]
            for col_channel in 1:N
                diff2 = data[col_channel, v, idx] - mean_vec[col_channel]
                cov_mat[row_channel, col_channel] += diff1 * diff2
            end
        end
    end
    cov_mat ./= (n_subjects - 1)
end

function compute_cov_trace_all_voxels(data::Array{T,3}, subject_indices::AbstractVector{<:Integer}) where T
    n_channels, n_voxels, n_subjects = size(data)
    variances = Array{T}(undef, n_voxels)
    
    mean_vec = @MVector zeros(T, 3)
    cov_mat = @MMatrix zeros(T, 3, 3)
    
    @inbounds for voxel_i in 1:n_voxels
        arr_mean!(mean_vec, data, voxel_i, n_subjects, subject_indices)
        fill!(cov_mat, zero(T))
        arr_covariance!(cov_mat, data, mean_vec, voxel_i, n_subjects, subject_indices)
        variances[voxel_i] = tr(cov_mat)
    end
    
    return variances
end


function voxel_covariances_op2(data::Array{T,3}) where T
    n_channels, n_voxels, n_subjects = size(data)
    variances = Array{T}(undef, n_voxels)
    mean_vec = Vector{T}(undef, n_channels)
    cov_mat = Matrix{T}(undef, n_channels, n_channels)
    zero_mat = zeros(T, n_channels, n_channels)

    for voxel_i in 1:n_voxels
        arr_mean!(mean_vec, data, voxel_i, n_subjects)
        arr_covariance!(cov_mat, data, mean_vec, voxel_i, n_subjects, n_channels, zero_mat)
        variances[voxel_i] = tr(cov_mat)
    end

    return variances
end


function voxel_covariances_static(data::Array{T,3}) where T
    n_channels, n_voxels, n_subjects = size(data)
    variances = Array{T}(undef, n_voxels)
    
    mean_vec = @MVector zeros(T, 3)
    cov_mat = @MMatrix zeros(T, 3, 3)
    
    for voxel_i in 1:n_voxels
        arr_mean!(mean_vec, data, voxel_i, n_subjects)
        fill!(cov_mat, zero(T))
        arr_covariance!(cov_mat, data, mean_vec, voxel_i, n_subjects)
        variances[voxel_i] = tr(cov_mat)
    end

    return variances
end

function voxel_covariances_static(data::Array{T,3}, subject_indices::AbstractVector{<:Integer}) where T
    n_channels, n_voxels, n_subjects = size(data)
    variances = Array{T}(undef, n_voxels)
    
    mean_vec = @MVector zeros(T, 3)
    cov_mat = @MMatrix zeros(T, 3, 3)
    
    for voxel_i in 1:n_voxels
        arr_mean!(mean_vec, data, voxel_i, n_subjects, subject_indices)
        fill!(cov_mat, zero(T))
        arr_covariance!(cov_mat, data, mean_vec, voxel_i, n_subjects, subject_indices)
        variances[voxel_i] = tr(cov_mat)
    end

    return variances
end

x = rand(Float32, 3, 1_000, 100);

@assert all(isapprox.(voxel_covariances(x), voxel_covariances_op(x), rtol=1e-5, atol=1e-6))
@assert all(isapprox.(voxel_covariances(x), voxel_covariances_op2(x), rtol=1e-5, atol=1e-6))
@assert all(isapprox.(voxel_covariances(x), voxel_covariances_static(x), rtol=1e-5, atol=1e-6))
@assert all(isapprox.(voxel_covariances(x), compute_cov_trace_all_voxels(x, collect(1:size(x, 3))), rtol=1e-5, atol=1e-6))
@time voxel_covariances(x);
@time voxel_covariances_op(x);
@time voxel_covariances_op2(x);
@time voxel_covariances_static(x);
@time compute_cov_trace_all_voxels(x, collect(1:size(x, 3)));
nothing

# 935.985 μs (12003 allocations: 1.85 MiB)
# 1.050 ms (7 allocations: 35.46 KiB)
# 927.389 μs (9 allocations: 35.57 KiB)
# 232.324 μs (5 allocations: 35.35 KiB)

# println("\n=== Native Assembly for 3x3 Static Array (your actual use case) ===")
# cov_mat3 = @MMatrix zeros(Float32, 3, 3)
# mean_vec3 = @MVector zeros(Float32, 3)
# data3 = rand(Float32, 3, 1000, 100)
# print(@code_native arr_covariance!(cov_mat3, data3, mean_vec3, 1, 100))