using Random
using ImageFiltering

function create_coordinate_grid_2d(dim_size, range_val)
    x = range(range_val[1], range_val[2]; length=dim_size)
    y = range(range_val[1], range_val[2]; length=dim_size)
    X = repeat(reshape(x, :, 1), 1, dim_size)
    Y = repeat(reshape(y, 1, :), dim_size, 1)
    return X, Y
end

function create_coordinate_grid_3d(dim_size, range_val)
    nx, ny, nz = dim_size
    x = range(range_val[1], range_val[2]; length=nx)
    y = range(range_val[1], range_val[2]; length=ny)
    z = range(range_val[1], range_val[2]; length=nz)
    X = repeat(reshape(x, :, 1, 1), 1, ny, nz)
    Y = repeat(reshape(y, 1, :, 1), nx, 1, nz)
    Z = repeat(reshape(z, 1, 1, :), nx, ny, 1)
    return X, Y, Z
end

function gaussian_peak_2d(X, Y, center, width, amplitude)
    return amplitude .* exp.(-((X .- center[1]).^2 .+ (Y .- center[2]).^2) ./ width)
end

function gaussian_peak_3d(X, Y, Z, center, width, amplitude)
    return amplitude .* exp.(-((X .- center[1]).^2 .+ (Y .- center[2]).^2 .+ (Z .- center[3]).^2) ./ width)
end

function anisotropic_gaussian_2d(X, Y, center, widths, angle, amplitude)
    cos_a, sin_a = cos(angle), sin(angle)
    X_rot = X .* cos_a .+ Y .* sin_a
    Y_rot = X .* sin_a .- Y .* cos_a
    return amplitude .* exp.(-((X_rot .- center[1]).^2 ./ widths[1] .+ 
                               (Y_rot .- center[2]).^2 ./ widths[2]))
end

function anisotropic_gaussian_3d_rotated_xy(X, Y, Z, center, widths_xy, width_z, angle, amplitude)
    cos_a, sin_a = cos(angle), sin(angle)
    X_rot = X .* cos_a .+ Y .* sin_a
    Y_rot = X .* sin_a .- Y .* cos_a
    return amplitude .* exp.(-((X_rot .- center[1]).^2 ./ widths_xy[1] .+ 
                               (Y_rot .- center[2]).^2 ./ widths_xy[2] .+
                               (Z .- center[3]).^2 ./ width_z))
end

function anisotropic_gaussian_3d_xz_transform(X, Y, Z, center_y, widths, amplitude)
    return amplitude .* exp.(-((X .+ Z).^2 ./ widths[1] .+ 
                              (Y .- center_y).^2 ./ widths[2] .+
                              (X .- Z).^2 ./ widths[3]))
end

function generate_2d_data(dim_size; seed=42, range_val=(-4, 4), noise=0.08, smooth_sigma=1.2)
    Random.seed!(seed)
    X, Y = create_coordinate_grid_2d(dim_size, range_val)

    Z = gaussian_peak_2d(X, Y, (2.5, 2.5), 1.0, 4.0) .+
        gaussian_peak_2d(X, Y, (-2.5, -2.5), 1.0, 3.5) .+
        gaussian_peak_2d(X, Y, (1.5, -2.0), 0.8, 3.0) .+
        gaussian_peak_2d(X, Y, (-2.0, 1.5), 0.6, 2.5) .+
        gaussian_peak_2d(X, Y, (0.5, 0.5), 1.5, 2.0) .+
        gaussian_peak_2d(X, Y, (-1.0, -1.0), 1.2, 1.8) .+
        gaussian_peak_2d(X, Y, (3.0, 0.5), 0.7, 2.3) .+
        gaussian_peak_2d(X, Y, (-3.0, -0.5), 0.9, 2.1) .+
        gaussian_peak_2d(X, Y, (0.8, 2.5), 0.85, 1.9) .+
        gaussian_peak_2d(X, Y, (-0.8, -2.5), 0.75, 1.7) .+
        gaussian_peak_2d(X, Y, (0.0, 0.0), 2.0, -1.5) .+
        gaussian_peak_2d(X, Y, (1.8, -1.2), 1.8, -1.2) .+
        gaussian_peak_2d(X, Y, (-1.5, 2.2), 1.5, -1.0) .+
        gaussian_peak_2d(X, Y, (2.2, 1.8), 1.3, -0.9) .+
        gaussian_peak_2d(X, Y, (-2.2, -1.8), 1.4, -0.8) .+
        anisotropic_gaussian_2d(X, Y, (0.0, 0.0), (0.5, 8.0), π/4, 2.2) .+
        anisotropic_gaussian_2d(X, Y, (0.0, 0.0), (0.6, 10.0), -π/6, 1.8) .+
        anisotropic_gaussian_2d(X, Y, (0.0, 0.0), (0.4, 12.0), π/3, 1.5) .+
        anisotropic_gaussian_2d(X, Y, (1.5, 1.5), (0.55, 9.0), π/8, 1.6) .+
        anisotropic_gaussian_2d(X, Y, (-1.5, -1.5), (0.65, 11.0), -π/4, 1.4) .+
        noise .* randn(size(X))

    return imfilter(Z, Kernel.gaussian(smooth_sigma))
end

function generate_3d_data(dim_size, seed=42, range_val=(-3, 3), noise=0.06, smooth_sigma=(1.0, 1.0, 1.0))
    Random.seed!(seed)
    X, Y, Z = create_coordinate_grid_3d(dim_size, range_val)

    data = gaussian_peak_3d(X, Y, Z, (1.5, 1.5, 1.5), 1.0, 3.0) .+
           gaussian_peak_3d(X, Y, Z, (-1.5, -1.5, -1.5), 1.0, 2.5) .+
           gaussian_peak_3d(X, Y, Z, (0.5, -1.0, 1.0), 0.8, 2.0) .+
           gaussian_peak_3d(X, Y, Z, (0.0, 0.0, 0.0), 1.5, 2.5) .+
           gaussian_peak_3d(X, Y, Z, (-0.8, 0.8, -0.8), 0.6, 2.2) .+
           gaussian_peak_3d(X, Y, Z, (2.0, -0.5, -2.0), 1.0, 1.8) .+
           gaussian_peak_3d(X, Y, Z, (-2.0, 2.0, 0.5), 0.9, 1.6) .+
           gaussian_peak_3d(X, Y, Z, (1.0, -1.5, -1.2), 1.2, -1.4) .+
           gaussian_peak_3d(X, Y, Z, (-1.2, 1.8, 1.5), 1.1, -1.2) .+
           anisotropic_gaussian_3d_rotated_xy(X, Y, Z, (0.0, 0.0, 0.0), (0.5, 8.0), 2.0, π/4, 1.8) .+
           anisotropic_gaussian_3d_xz_transform(X, Y, Z, 0.5, (0.7, 10.0, 12.0), 1.5) .+
           noise .* randn(size(X))

    return imfilter(data, Kernel.gaussian(smooth_sigma))
end

