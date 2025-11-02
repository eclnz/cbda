include("coord.jl")

using Test

@testset "Coord Projection Tests" begin
    
    @testset "Coord1D project_ray" begin
        data = [1.0, 2.0, 0.0, 4.0, 5.0]
        
        @testset "Find zero value" begin
            coord = Coord1D(1)
            result = project_ray(data, coord, 1)
            @test result.x == 3
        end
        
        @testset "Move right, hit boundary" begin
            coord = Coord1D(4)
            result = project_ray(data, coord, 1)
            @test result.x == 5
        end
        
        @testset "Move left, hit boundary" begin
            coord = Coord1D(2)
            result = project_ray(data, coord, -1)
            @test result.x == 1
        end
        
        @testset "Start at left boundary, move left" begin
            coord = Coord1D(1)
            result = project_ray(data, coord, -1)
            @test result.x == 1
        end
        
        @testset "Start at right boundary, move right" begin
            coord = Coord1D(5)
            result = project_ray(data, coord, 1)
            @test result.x == 5
        end
        
        @testset "Zero at boundary" begin
            data_boundary = [0.0, 1.0, 2.0]
            coord = Coord1D(2)
            result = project_ray(data_boundary, coord, -1)
            @test result.x == 1
        end
        
        @testset "No zeros, hit boundary" begin
            data_no_zero = [1.0, 2.0, 3.0]
            coord = Coord1D(2)
            result = project_ray(data_no_zero, coord, 1)
            @test result.x == 3
        end
    end
    
    @testset "Coord2D project_ray" begin
        data = [1.0 2.0 3.0
                4.0 0.0 6.0
                7.0 8.0 9.0]
        
        @testset "Find zero value" begin
            coord = Coord2D(1, 1)
            direction = Coord2D(1, 1)
            result = project_ray(data, coord, direction)
            @test result.x == 2
            @test result.y == 2
        end
        
        @testset "Move right, hit boundary" begin
            coord = Coord2D(1, 2)
            direction = Coord2D(0, 1)
            result = project_ray(data, coord, direction)
            @test result.x == 1
            @test result.y == 3
        end
        
        @testset "Move left, hit boundary" begin
            coord = Coord2D(2, 2)
            direction = Coord2D(-1, 0)
            result = project_ray(data, coord, direction)
            @test result.x == 1
            @test result.y == 2
        end
        
        @testset "Move up, hit boundary" begin
            coord = Coord2D(2, 2)
            direction = Coord2D(-1, 0)
            result = project_ray(data, coord, direction)
            @test result.x == 1
            @test result.y == 2
        end
        
        @testset "Move down, hit boundary" begin
            coord = Coord2D(2, 2)
            direction = Coord2D(1, 0)
            result = project_ray(data, coord, direction)
            @test result.x == 3
            @test result.y == 2
        end
        
        @testset "Diagonal movement" begin
            coord = Coord2D(1, 1)
            direction = Coord2D(1, 1)
            result = project_ray(data, coord, direction)
            @test result.x == 2
            @test result.y == 2
        end
        
        @testset "Start at corner, move diagonal" begin
            coord = Coord2D(1, 1)
            direction = Coord2D(-1, -1)
            result = project_ray(data, coord, direction)
            @test result.x == 1
            @test result.y == 1
        end
        
        @testset "Move to opposite corner" begin
            coord = Coord2D(1, 1)
            direction = Coord2D(1, 1)
            result = project_ray(data, coord, direction)
            @test 1 ≤ result.x ≤ 3
            @test 1 ≤ result.y ≤ 3
        end
        
        @testset "Zero at boundary" begin
            data_boundary = [0.0 1.0
                             2.0 3.0]
            coord = Coord2D(1, 2)
            direction = Coord2D(-1, 0)
            result = project_ray(data_boundary, coord, direction)
            @test result.x == 1
            @test result.y == 2
        end
    end
    
    @testset "Coord3D project_ray" begin
        data = cat([1.0 2.0 3.0
                    4.0 5.0 6.0
                    7.0 8.0 9.0],
                   [10.0 11.0 12.0
                    13.0 0.0 15.0
                    16.0 17.0 18.0],
                   [19.0 20.0 21.0
                    22.0 23.0 24.0
                    25.0 26.0 27.0], dims=3)
        
        @testset "Find zero value" begin
            coord = Coord3D(1, 1, 1)
            direction = Coord3D(1, 1, 1)
            result = project_ray(data, coord, direction)
            @test result.x == 2
            @test result.y == 2
            @test result.z == 2
        end
        
        @testset "Move in x direction, hit boundary" begin
            coord = Coord3D(2, 2, 2)
            direction = Coord3D(1, 0, 0)
            result = project_ray(data, coord, direction)
            @test result.x == 3
            @test result.y == 2
            @test result.z == 2
        end
        
        @testset "Move in y direction, hit boundary" begin
            coord = Coord3D(2, 2, 2)
            direction = Coord3D(0, 1, 0)
            result = project_ray(data, coord, direction)
            @test result.x == 2
            @test result.y == 3
            @test result.z == 2
        end
        
        @testset "Move in z direction, hit boundary" begin
            coord = Coord3D(2, 2, 2)
            direction = Coord3D(0, 0, 1)
            result = project_ray(data, coord, direction)
            @test result.x == 2
            @test result.y == 2
            @test result.z == 3
        end
        
        @testset "Move negative direction" begin
            coord = Coord3D(2, 2, 2)
            direction = Coord3D(-1, 0, 0)
            result = project_ray(data, coord, direction)
            @test result.x == 1
            @test result.y == 2
            @test result.z == 2
        end
        
        @testset "Start at corner" begin
            coord = Coord3D(1, 1, 1)
            direction = Coord3D(-1, -1, -1)
            result = project_ray(data, coord, direction)
            @test result.x == 1
            @test result.y == 1
            @test result.z == 1
        end
        
        @testset "Diagonal movement" begin
            coord = Coord3D(1, 1, 1)
            direction = Coord3D(1, 1, 1)
            result = project_ray(data, coord, direction)
            @test result.x == 2
            @test result.y == 2
            @test result.z == 2
        end
        
        @testset "All coordinates within bounds" begin
            coord = Coord3D(2, 2, 2)
            direction = Coord3D(1, 1, 1)
            result = project_ray(data, coord, direction)
            @test 1 ≤ result.x ≤ 3
            @test 1 ≤ result.y ≤ 3
            @test 1 ≤ result.z ≤ 3
        end
    end
    
    @testset "project_all_rays" begin
        @testset "Coord1D project_all_rays" begin
            data = [1.0, 0.0, 3.0]
            coord = Coord1D(2)
            results = project_all_rays(data, coord)
            @test length(results) == 2
            @test results[1].x ∈ [1, 3]
            @test results[2].x ∈ [1, 3]
            @test results[1].x != results[2].x
        end
        
        @testset "Coord1D project_all_rays at boundary" begin
            data = [1.0, 2.0, 3.0]
            coord = Coord1D(1)
            results = project_all_rays(data, coord)
            @test length(results) == 2
        end
        
        @testset "Coord2D project_all_rays" begin
            data = [1.0 2.0 3.0
                    4.0 0.0 6.0
                    7.0 8.0 9.0]
            coord = Coord2D(2, 2)
            results = project_all_rays(data, coord)
            @test length(results) == 8
            for result in results
                @test 1 ≤ result.x ≤ 3
                @test 1 ≤ result.y ≤ 3
            end
        end
        
        @testset "Coord2D project_all_rays at corner" begin
            data = [1.0 2.0
                    3.0 4.0]
            coord = Coord2D(1, 1)
            results = project_all_rays(data, coord)
            @test length(results) == 8
            for result in results
                @test 1 ≤ result.x ≤ 2
                @test 1 ≤ result.y ≤ 2
            end
        end
        
        @testset "Coord3D project_all_rays" begin
            data = cat([1.0 2.0
                        3.0 4.0],
                       [5.0 6.0
                        7.0 8.0], dims=3)
            coord = Coord3D(1, 1, 1)
            results = project_all_rays(data, coord)
            @test length(results) == 26
            for result in results
                @test 1 ≤ result.x ≤ 2
                @test 1 ≤ result.y ≤ 2
                @test 1 ≤ result.z ≤ 2
            end
        end
        
        @testset "Coord3D project_all_rays at center" begin
            data = cat([1.0 2.0 3.0
                        4.0 5.0 6.0
                        7.0 8.0 9.0],
                       [10.0 11.0 12.0
                        13.0 14.0 15.0
                        16.0 17.0 18.0],
                       [19.0 20.0 21.0
                        22.0 23.0 24.0
                        25.0 26.0 27.0], dims=3)
            coord = Coord3D(2, 2, 2)
            results = project_all_rays(data, coord)
            @test length(results) == 26
            for result in results
                @test 1 ≤ result.x ≤ 3
                @test 1 ≤ result.y ≤ 3
                @test 1 ≤ result.z ≤ 3
            end
        end
    end
    
    @testset "Edge cases" begin
        @testset "Single element array 1D" begin
            data = [0.0]
            coord = Coord1D(1)
            result = project_ray(data, coord, 1)
            @test result.x == 1
        end
        
        @testset "Single element array 2D" begin
            data = [0.0;;]
            coord = Coord2D(1, 1)
            direction = Coord2D(1, 1)
            result = project_ray(data, coord, direction)
            @test result.x == 1
            @test result.y == 1
        end
        
        @testset "All zeros" begin
            data = [0.0, 0.0, 0.0]
            coord = Coord1D(2)
            result = project_ray(data, coord, 1)
            @test result.x == 3
        end
        
        @testset "No zeros" begin
            data = [1.0, 2.0, 3.0]
            coord = Coord1D(1)
            result = project_ray(data, coord, 1)
            @test result.x == 3
        end
        
        @testset "Custom epsval" begin
            data = [0.1, 0.01, 0.001]
            coord = Coord1D(1)
            result = project_ray(data, coord, 1; epsval=0.01)
            @test result.x == 3
        end
        
        @testset "Very small values" begin
            data = [1e-10, 1e-20, 1e-30]
            coord = Coord1D(1)
            result = project_ray(data, coord, 1; epsval=1e-15)
            @test result.x == 2
        end
    end
    
    @testset "unit_directions dispatch" begin
        @testset "Coord1D directions" begin
            dirs = collect(unit_directions(Coord1D))
            @test length(dirs) == 2
            @test -1 in dirs
            @test 1 in dirs
        end
        
        @testset "Coord2D directions" begin
            dirs = collect(unit_directions(Coord2D))
            @test length(dirs) == 8
            for dir in dirs
                @test dir.x ∈ [-1, 0, 1]
                @test dir.y ∈ [-1, 0, 1]
                @test !(dir.x == 0 && dir.y == 0)
            end
        end
        
        @testset "Coord3D directions" begin
            dirs = collect(unit_directions(Coord3D))
            @test length(dirs) == 26
            for dir in dirs
                @test dir.x ∈ [-1, 0, 1]
                @test dir.y ∈ [-1, 0, 1]
                @test dir.z ∈ [-1, 0, 1]
                @test !(dir.x == 0 && dir.y == 0 && dir.z == 0)
            end
        end
    end
end

nothing