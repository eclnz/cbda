include("conv.jl")

using BenchmarkTools
using Random

println("="^70)
println("Convolution Benchmarks")
println("="^70)

Random.seed!(42)

# Warmup
println("\nWarming up...")
warmup_data_1d = rand(100)
warmup_kernel = gaussian_kernel(21, 2.5)
warmup_result = conv_valid(warmup_data_1d, warmup_kernel)

# Benchmark parameters
kernel_sizes = [5, 11, 21]
data_sizes_1d = [100, 500, 1000, 5000]
data_sizes_2d = [(50, 50), (100, 100), (200, 200)]
data_sizes_3d = [(20, 20, 20), (30, 30, 30), (50, 50, 50)]

println("\n" * "="^70)
println("1D Convolution Benchmarks")
println("="^70)

for size_val in data_sizes_1d
    data = rand(size_val)
    kernel = gaussian_kernel(21, 2.5)
    
    println("\nData size: $size_val")
    b = @benchmark conv_valid($data, $kernel)
    println("  Time: $(round(median(b.times) / 1e6, digits=2)) ms")
    println("  Memory: $(round(b.memory / 1024, digits=2)) KB")
    println("  Allocations: $(b.allocs)")
end

println("\n" * "="^70)
println("2D Convolution Benchmarks")
println("="^70)

for (n1, n2) in data_sizes_2d
    data = rand(n1, n2)
    kernel = gaussian_kernel(21, 2.5)
    
    println("\nData size: ($n1, $n2)")
    b = @benchmark conv_valid($data, $kernel)
    println("  Time: $(round(median(b.times) / 1e6, digits=2)) ms")
    println("  Memory: $(round(b.memory / 1024, digits=2)) KB")
    println("  Allocations: $(b.allocs)")
end

println("\n" * "="^70)
println("3D Convolution Benchmarks")
println("="^70)

for (n1, n2, n3) in data_sizes_3d
    data = rand(n1, n2, n3)
    kernel = gaussian_kernel(21, 2.5)
    
    println("\nData size: ($n1, $n2, $n3)")
    b = @benchmark conv_valid($data, $kernel)
    println("  Time: $(round(median(b.times) / 1e6, digits=2)) ms")
    println("  Memory: $(round(b.memory / 1024, digits=2)) KB")
    println("  Allocations: $(b.allocs)")
end

println("\n" * "="^70)
println("Kernel Size Comparison (1D)")
println("="^70)

data = rand(1000)
for ksize in kernel_sizes
    kernel = gaussian_kernel(ksize, 2.5)
    println("\nKernel size: $ksize")
    b = @benchmark conv_valid($data, $kernel)
    println("  Time: $(round(median(b.times) / 1e6, digits=2)) ms")
    println("  Memory: $(round(b.memory / 1024, digits=2)) KB")
end

println("\n" * "="^70)
println("process_signal Benchmarks")
println("="^70)

println("\n1D:")
data = randn(500)
b = @benchmark process_signal($data)
println("  Time: $(round(median(b.times) / 1e6, digits=2)) ms")
println("  Memory: $(round(b.memory / 1024, digits=2)) KB")

println("\n2D:")
data = randn(100, 100)
b = @benchmark process_signal($data)
println("  Time: $(round(median(b.times) / 1e6, digits=2)) ms")
println("  Memory: $(round(b.memory / 1024, digits=2)) KB")

println("\n3D:")
data = randn(50, 50, 50)
b = @benchmark process_signal($data)
println("  Time: $(round(median(b.times) / 1e6, digits=2)) ms")
println("  Memory: $(round(b.memory / 1024, digits=2)) KB")

println("\n" * "="^70)
println("conv_along_dim! Benchmarks")
println("="^70)

println("\n2D - Dimension 1:")
data = rand(200, 200)
kernel = gaussian_kernel(21, 2.5)
output = similar(data, size(data, 1) - length(kernel) + 1, size(data, 2))
b = @benchmark conv_along_dim!($output, $data, $kernel, 1)
println("  Time: $(round(median(b.times) / 1e6, digits=2)) ms")
println("  Memory: $(round(b.memory / 1024, digits=2)) KB")

println("\n2D - Dimension 2:")
data = rand(200, 200)
kernel = gaussian_kernel(21, 2.5)
output = similar(data, size(data, 1), size(data, 2) - length(kernel) + 1)
b = @benchmark conv_along_dim!($output, $data, $kernel, 2)
println("  Time: $(round(median(b.times) / 1e6, digits=2)) ms")
println("  Memory: $(round(b.memory / 1024, digits=2)) KB")

println("\n3D - Dimension 1:")
data = rand(50, 50, 50)
kernel = gaussian_kernel(21, 2.5)
output = similar(data, size(data, 1) - length(kernel) + 1, size(data, 2), size(data, 3))
b = @benchmark conv_along_dim!($output, $data, $kernel, 1)
println("  Time: $(round(median(b.times) / 1e6, digits=2)) ms")
println("  Memory: $(round(b.memory / 1024, digits=2)) KB")

println("\n" * "="^70)
println("Benchmarks complete!")
println("="^70)

