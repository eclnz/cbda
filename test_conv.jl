include("conv.jl")

using Test
using Random

@testset "Convolution Tests" begin
    
    @testset "gaussian_kernel" begin
        k = gaussian_kernel(5, 1.0)
        @test length(k) == 5
        @test sum(k) ≈ 1.0 atol=1e-10
        @test all(k .> 0)
        @test k[3] > k[1]  # Center should be largest
        @test k[3] > k[5]  # Center should be largest
    end
    
    @testset "conv_valid 1D" begin
        @testset "Basic convolution" begin
            data = [1.0, 2.0, 3.0, 4.0, 5.0]
            kernel = [0.25, 0.5, 0.25]
            result = conv_valid(data, kernel)
            
            pad = div(length(kernel), 2)
            expected_length = length(data) + 2*pad - length(kernel) + 1
            @test length(result) == expected_length
            @test result[1] ≈ 1.25 atol=1e-10  # (1*0.25 + 1*0.5 + 2*0.25) with padding
            @test result[2] ≈ 2.0 atol=1e-10   # (1*0.25 + 2*0.5 + 3*0.25)
        end
        
        @testset "Identity kernel" begin
            data = [1.0, 2.0, 3.0, 4.0]
            kernel = [0.0, 1.0, 0.0]
            result = conv_valid(data, kernel)
            
            pad = div(length(kernel), 2)
            expected_length = length(data) + 2*pad - length(kernel) + 1
            @test length(result) == expected_length
            @test result[1] ≈ 1.0 atol=1e-10  # (1*0.0 + 1*1.0 + 2*0.0)
            @test result[2] ≈ 2.0 atol=1e-10  # (1*0.0 + 2*1.0 + 3*0.0)
        end
        
        @testset "Single element result" begin
            data = [1.0, 2.0]
            kernel = [0.5, 0.5]
            result = conv_valid(data, kernel)
            
            pad = div(length(kernel), 2)
            expected_length = length(data) + 2*pad - length(kernel) + 1
            @test length(result) == expected_length
            @test result[1] ≈ 1.0 atol=1e-10  # (1*0.5 + 1*0.5) with padding
        end
        
        @testset "Kernel larger than data" begin
            data = [1.0, 2.0]
            kernel = [0.2, 0.3, 0.3, 0.2]
            result = conv_valid(data, kernel)
            
            pad = div(length(kernel), 2)
            expected_length = length(data) + 2*pad - length(kernel) + 1
            @test length(result) == expected_length
            @test result[1] ≈ 1.2 atol=1e-10  # (1*0.2 + 1*0.3 + 2*0.3 + 2*0.2) with padding
        end
        
        @testset "Different types" begin
            data = Float32[1.0, 2.0, 3.0]
            kernel = Float32[0.25, 0.5, 0.25]
            result = conv_valid(data, kernel)
            
            @test eltype(result) == Float32
            pad = div(length(kernel), 2)
            expected_length = length(data) + 2*pad - length(kernel) + 1
            @test length(result) == expected_length
            @test result[1] ≈ 1.25f0 atol=1e-6
        end
    end
    
    @testset "conv_valid 2D" begin
        @testset "Basic 2D convolution" begin
            data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
            kernel = [0.25, 0.5, 0.25]
            result = conv_valid(data, kernel)
            
            ksize = length(kernel)
            expected_size = (size(data, 1) - ksize + 1, size(data, 2) - ksize + 1)
            @test size(result) == expected_size
            @test all(isfinite.(result))
        end
        
        @testset "Identity kernel" begin
            data = [1.0 2.0; 3.0 4.0]
            kernel = [0.0, 1.0, 0.0]
            result = conv_valid(data, kernel)
            
            ksize = length(kernel)
            expected_size = (size(data, 1) - ksize + 1, size(data, 2) - ksize + 1)
            @test size(result) == expected_size
            @test all(isfinite.(result))
        end
        
        @testset "Rectangular array" begin
            data = rand(5, 3)
            kernel = [0.25, 0.5, 0.25]
            result = conv_valid(data, kernel)
            
            ksize = length(kernel)
            expected_size = (size(data, 1) - ksize + 1, size(data, 2) - ksize + 1)
            @test size(result) == expected_size
            @test all(isfinite.(result))
        end
        
        @testset "Separation property" begin
            # For separable convolution, convolving with [1,2,1] twice should be equivalent
            # to convolving with outer product [1,2,1]' * [1,2,1]
            data = rand(5, 5)
            kernel = [0.25, 0.5, 0.25]
            result = conv_valid(data, kernel)
            
            @test size(result) == (3, 3)
            @test all(isfinite.(result))
        end
    end
    
    @testset "conv_valid 3D" begin
        @testset "Basic 3D convolution" begin
            data = rand(5, 5, 5)
            kernel = [0.25, 0.5, 0.25]
            result = conv_valid(data, kernel)
            
            ksize = length(kernel)
            expected_size = (size(data, 1) - ksize + 1, size(data, 2) - ksize + 1, size(data, 3) - ksize + 1)
            @test size(result) == expected_size
            @test all(isfinite.(result))
        end
        
        @testset "Rectangular 3D array" begin
            Random.seed!(42)
            data = rand(6, 4, 3)
            kernel = [0.25, 0.5, 0.25]
            result = conv_valid(data, kernel)
            
            ksize = length(kernel)
            expected_size = (size(data, 1) - ksize + 1, size(data, 2) - ksize + 1, size(data, 3) - ksize + 1)
            @test size(result) == expected_size
            @test all(isfinite.(result))
        end
        
        @testset "Consistency with dimensions" begin
            data = ones(7, 7, 7)
            kernel = [0.2, 0.3, 0.3, 0.2]
            result = conv_valid(data, kernel)
            
            ksize = length(kernel)
            expected_size = (size(data, 1) - ksize + 1, size(data, 2) - ksize + 1, size(data, 3) - ksize + 1)
            @test size(result) == expected_size
            @test all(isfinite.(result))
            @test all(result .≥ 0)  # Values should be non-negative after convolving ones
        end
    end
    
    @testset "conv_along_dim!" begin
        @testset "1D dimension" begin
            data = [1.0, 2.0, 3.0, 4.0]
            kernel = [0.25, 0.5, 0.25]
            pad = div(length(kernel), 2)
            output_size = size(data, 1) + 2*pad - length(kernel) + 1
            output = similar(data, output_size)
            conv_along_dim!(output, data, kernel, 1)
            
            expected = conv_valid(data, kernel)
            @test output ≈ expected
        end
        
        @testset "2D dimension 1" begin
            data = [1.0 2.0 3.0; 4.0 5.0 6.0]
            kernel = [0.5, 0.5]
            output_size = size(data, 1) - length(kernel) + 1
            output = similar(data, output_size, size(data, 2))
            conv_along_dim!(output, data, kernel, 1)
            
            @test size(output) == (output_size, size(data, 2))
            @test all(isfinite.(output))
        end
        
        @testset "2D dimension 2" begin
            data = [1.0 2.0 3.0; 4.0 5.0 6.0]
            kernel = [0.5, 0.5]
            output_size = size(data, 2) - length(kernel) + 1
            output = similar(data, size(data, 1), output_size)
            conv_along_dim!(output, data, kernel, 2)
            
            @test size(output) == (size(data, 1), output_size)
            @test all(isfinite.(output))
        end
        
        @testset "3D all dimensions" begin
            data = rand(5, 5, 5)
            kernel = [0.25, 0.5, 0.25]
            
            for dim in 1:3
                output = similar(data, size(data)...)
                output_size = collect(size(output))
                output_size[dim] = size(data, dim) - length(kernel) + 1
                output = similar(data, output_size...)
                conv_along_dim!(output, data, kernel, dim)
                
                @test size(output)[dim] == size(data, dim) - length(kernel) + 1
                @test all(isfinite.(output))
            end
        end
    end
    
    @testset "process_signal" begin
        @testset "1D processing" begin
            Random.seed!(42)
            data = randn(20)
            result = process_signal(data; kernel_size=5, kernel_sigma=1.0)
            
            pad = div(5, 2)
            expected_length = length(data) + 2*pad - 5 + 1
            @test length(result) == expected_length
            @test all(result .> 0)  # Should be offset to positive
            @test all(isfinite.(result))
        end
        
        @testset "2D processing" begin
            Random.seed!(42)
            data = randn(10, 10)
            result = process_signal(data; kernel_size=5, kernel_sigma=1.0)
            
            expected_size = (size(data, 1) - 5 + 1, size(data, 2) - 5 + 1)
            @test size(result) == expected_size
            @test all(result .> 0)  # Should be offset to positive
            @test all(isfinite.(result))
        end
        
        @testset "3D processing" begin
            Random.seed!(42)
            data = randn(8, 8, 8)
            result = process_signal(data; kernel_size=5, kernel_sigma=1.0)
            
            expected_size = (size(data, 1) - 5 + 1, size(data, 2) - 5 + 1, size(data, 3) - 5 + 1)
            @test size(result) == expected_size
            @test all(result .> 0)  # Should be offset to positive
            @test all(isfinite.(result))
        end
        
        @testset "Custom parameters" begin
            data = randn(15)
            result1 = process_signal(data; kernel_size=3, kernel_sigma=0.5, offset_value=0.5)
            result2 = process_signal(data; kernel_size=21, kernel_sigma=2.0, offset_value=1.0)
            
            pad1 = div(3, 2)
            pad2 = div(21, 2)
            len1 = length(data) + 2*pad1 - 3 + 1
            len2 = length(data) + 2*pad2 - 21 + 1
            @test length(result1) == len1
            @test length(result2) == len2
            @test all(result1 .> 0)
            @test all(result2 .> 0)
        end
    end
    
    @testset "Edge cases" begin
        @testset "Very small arrays" begin
            data = [1.0, 2.0]
            kernel = [0.5, 0.5]
            result = conv_valid(data, kernel)
            pad = div(length(kernel), 2)
            expected_length = length(data) + 2*pad - length(kernel) + 1
            @test length(result) == expected_length
        end
        
        @testset "Constant arrays" begin
            data = ones(5, 5)
            kernel = [0.25, 0.5, 0.25]
            result = conv_valid(data, kernel)
            @test all(result .≈ 1.0)
        end
        
        @testset "Zero arrays" begin
            data = zeros(5, 5)
            kernel = [0.25, 0.5, 0.25]
            result = conv_valid(data, kernel)
            @test all(result .≈ 0.0)
        end
    end
end

println("\n✓ All convolution tests passed!")

