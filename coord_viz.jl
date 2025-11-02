include("coord.jl")
using Plots
using Random

function generate_smooth_1d(size)
    x = range(0, 2π, length=size)
    data = sin.(x) .+ 0.5 .* sin.(3 .* x) .+ 0.3 .* randn(size)
    return data
end

function generate_smooth_2d(size)
    x = range(0, 2π, length=size)
    y = range(0, 2π, length=size)
    data = zeros(size, size)
    for i in 1:size
        for j in 1:size
            data[i, j] = sin(x[i]) * cos(y[j]) + 0.2 * randn()
        end
    end
    return data
end

function generate_smooth_3d(size)
    x = range(0, 2π, length=size)
    y = range(0, 2π, length=size)
    z = range(0, 2π, length=size)
    data = zeros(size, size, size)
    for i in 1:size
        for j in 1:size
            for k in 1:size
                data[i, j, k] = sin(x[i]) * cos(y[j]) * sin(z[k]) + 0.2 * randn()
            end
        end
    end
    return data
end

function visualize_all(; data_size_1d=50, data_size_2d=30, data_size_3d=15, seed=42)
    Random.seed!(seed)
    
    data1d = generate_smooth_1d(data_size_1d)
    coord1d = Coord1D(data_size_1d ÷ 2)
    results1d = project_all_rays(data1d, coord1d)
    
    data2d = generate_smooth_2d(data_size_2d)
    coord2d = Coord2D(data_size_2d ÷ 2, data_size_2d ÷ 2)
    results2d = project_all_rays(data2d, coord2d)
    
    data3d = generate_smooth_3d(data_size_3d)
    coord3d = Coord3D(data_size_3d ÷ 2, data_size_3d ÷ 2, data_size_3d ÷ 2)
    results3d = project_all_rays(data3d, coord3d)
    
    p1 = plot(data1d, lw=2, label="Data", xlabel="Index", ylabel="Value", title="1D Ray Projection")
    vline!(p1, [coord1d.x], color=:red, linestyle=:dash, label="Start", lw=2)
    for result in results1d
        scatter!(p1, [result.x], [data1d[result.x]], color=:blue, markersize=6, label=result == results1d[1] ? "Projected" : false)
    end
    
    p2 = heatmap(data2d, title="2D Ray Projection", xlabel="X", ylabel="Y", colorbar_title="Value")
    scatter!(p2, [coord2d.x], [coord2d.y], color=:red, markersize=10, label="Start", marker=:star)
    for result in results2d
        scatter!(p2, [result.x], [result.y], color=:blue, markersize=5, alpha=0.7, label=result == results2d[1] ? "Projected" : false)
    end
    for result in results2d
        plot!(p2, [coord2d.x, result.x], [coord2d.y, result.y], color=:green, alpha=0.2, linestyle=:dash, linewidth=1, label=result == results2d[1] ? "Rays" : false)
    end
    
    slice_xy = data3d[:, :, coord3d.z]
    slice_xz = data3d[:, coord3d.y, :]
    slice_yz = data3d[coord3d.x, :, :]
    
    p3 = heatmap(slice_xy, title="XY Slice (z=$(coord3d.z))", xlabel="X", ylabel="Y")
    scatter!(p3, [coord3d.x], [coord3d.y], color=:red, markersize=10, marker=:star, label="Start")
    for result in results3d
        if result.z == coord3d.z
            scatter!(p3, [result.x], [result.y], color=:blue, markersize=5, alpha=0.7, label=result == results3d[1] ? "Projected" : false)
        end
    end
    
    p4 = heatmap(slice_xz, title="XZ Slice (y=$(coord3d.y))", xlabel="X", ylabel="Z")
    scatter!(p4, [coord3d.x], [coord3d.z], color=:red, markersize=10, marker=:star, label="Start")
    for result in results3d
        if result.y == coord3d.y
            scatter!(p4, [result.x], [result.z], color=:blue, markersize=5, alpha=0.7, label=result == results3d[1] ? "Projected" : false)
        end
    end
    
    p5 = heatmap(slice_yz, title="YZ Slice (x=$(coord3d.x))", xlabel="Y", ylabel="Z")
    scatter!(p5, [coord3d.y], [coord3d.z], color=:red, markersize=10, marker=:star, label="Start")
    for result in results3d
        if result.x == coord3d.x
            scatter!(p5, [result.y], [result.z], color=:blue, markersize=5, alpha=0.7, label=result == results3d[1] ? "Projected" : false)
        end
    end
    
    p3d = plot(p3, p4, p5, layout=@layout([a b c]), size=(1400, 400))
    
    layout = @layout [
        a{0.25h}
        b{0.4h}
        c{0.35h}
    ]
    fig = plot(p1, p2, p3d, layout=layout, size=(1400, 1600))
    
    display(fig)
    gui()
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    visualize_all()
    println("\nPlots are open. Close the plot window to exit.")
    while true
        sleep(1)
    end
end

