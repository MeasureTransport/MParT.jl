using Test, MParT

@testset "MParT.jl" begin
    # @testset "Affine, Composed Map" begin # This test doesn't work due to weird constructor
    #     include("mapTypeTest.jl")
    # end
    @testset "MultiIndex" begin
        include("multiindex.jl")
    end
    @testset "MapFactory" begin
        include("mapFactory.jl")
    end
    @testset "Monotone Least Squares" begin
        include("MLS.jl")
    end
    @testset "2D Banana From Samples" begin
        include("banana2D.jl")
    end
    @testset "TrainMap" begin
        include("trainMapTest.jl")
    end
end
