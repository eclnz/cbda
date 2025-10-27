using StaticArrays
using LinearAlgebra
using LoopVectorization
using Random
using BenchmarkTools
using Plots

include("cov.jl")

function create_two_group_data(n_channels, n_voxels, n_subjects; group_diff=0.1f0, seed=42)
    Random.seed!(seed)
    data = rand(Float32, n_channels, n_voxels, n_subjects)
    
    n_group1 = n_subjects ÷ 2
    
    for v in 1:n_voxels
        for ch in 1:n_channels
            for s in (n_group1+1):n_subjects
                data[ch, v, s] += group_diff
            end
        end
    end
    
    return data, [ones(Int, n_group1); 2*ones(Int, n_subjects - n_group1)]
end

function two_group_permutation_test(data::Array{T,3}, group_labels::Vector{Int}, n_permutations::Int; seed::Int=42) where T
    n_channels, n_voxels, n_subjects = size(data)
    
    group1_indices = findall(x -> x == 1, group_labels)
    group2_indices = findall(x -> x == 2, group_labels)
    
    stats1 = voxel_covariances_static(data[:, :, group1_indices])
    stats2 = voxel_covariances_static(data[:, :, group2_indices])
    
    observed_diff = stats1 .- stats2
    permuted_diffs = Array{T}(undef, n_permutations, n_voxels)
    
    permutation = collect(1:n_subjects)
    Random.seed!(seed)
    
    for perm_idx in 1:n_permutations
        shuffle!(permutation)
        
        perm_group1 = collect(1:length(group1_indices))
        perm_group2 = collect((length(group1_indices)+1):n_subjects)
        
        stats1_perm = voxel_covariances_static(data[:, :, permutation[perm_group1]])
        stats2_perm = voxel_covariances_static(data[:, :, permutation[perm_group2]])
        
        permuted_diffs[perm_idx, :] = stats1_perm .- stats2_perm
    end
    
    pvals = zeros(T, n_voxels)
    for v in 1:n_voxels
        extreme_count = sum(abs.(permuted_diffs[:, v]) .>= abs(observed_diff[v]))
        pvals[v] = extreme_count / n_permutations
    end
    
    return observed_diff, permuted_diffs, pvals
end

x, group_labels = create_two_group_data(3, 1_000, 100, group_diff=0.15f0)

println("=== Two-group permutation test ===")
observed_diff, permuted_diffs, pvals = two_group_permutation_test(x, group_labels, 100)

println("\nTop voxels by effect:")
top_indices = sortperm(abs.(observed_diff), rev=true)
for i in 1:min(10, length(top_indices))
    v = top_indices[i]
    println("Voxel $v: diff=$(round(observed_diff[v], digits=3)), p=$(round(pvals[v], digits=3))")
end

println("\n=== Creating plots ===")
p = plot_permutation_results(observed_diff, permuted_diffs, pvals, top_indices[1])
display(p)