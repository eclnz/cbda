using ImageMorphology
using LoopVectorization

function tfce_like(Z::AbstractArray; E=0.5, H=2.0, nsteps=50)
    tfce = zeros(Float64, size(Z))
    z_min = minimum(Z)
    z_max = maximum(Z)
    levels = range(z_min, stop=z_max, length=nsteps)
    for h in 2:length(levels)
        lvl = levels[h]
        flooded = Z .>= lvl
        labeled = label_components(flooded)
        n_clusters = maximum(labeled)
        for label_id in 1:n_clusters
            cluster_mask = labeled .== label_id
            sz = count(cluster_mask)
            tfce[cluster_mask] .+= (sz^E) * ((lvl - z_min)^H)
        end
    end
    return tfce
end

function tfce_like!(tfce::AbstractArray, Z::AbstractArray, flooded::AbstractArray{Bool}, labeled::AbstractArray{Int}; E=0.5, H=2.0, nsteps=50)
    fill!(tfce, 0.0)
    z_min = minimum(Z)
    z_max = maximum(Z)
    levels = collect(range(z_min, stop=z_max, length=nsteps))
    
    for h in 2:length(levels)
        lvl = levels[h]
        @inbounds @turbo for i in eachindex(Z)
            flooded[i] = Z[i] >= lvl
        end
        
        label_components!(labeled, flooded)
        
        n_clusters = maximum(labeled)
        for label_id in 1:n_clusters
            sz = 0
            @inbounds for i in eachindex(labeled)
                if labeled[i] == label_id
                    sz += 1
                end
            end
            if sz == 0
                continue
            end
            incr = (sz^E) * ((lvl - z_min)^H)
            @inbounds @turbo for i in eachindex(tfce)
                tfce[i] += incr * (labeled[i] == label_id)
            end
        end
    end
    return tfce
end

function flood_fill_levels(Z::AbstractArray; nsteps=6)
    z_min = minimum(Z)
    z_max = maximum(Z)
    levels = collect(range(z_min, stop=z_max, length=nsteps))
    masks = Vector{Matrix{Int}}(undef, nsteps)
    for (idx, lvl) in enumerate(levels)
        flooded = Z .>= lvl
        labeled = label_components(flooded)
        masks[idx] = labeled
    end
    return levels, masks
end
