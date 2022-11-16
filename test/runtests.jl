using Test

@testset "MParT.jl" begin
    @testset "Affine, Composed Map" begin
        include("mapTypeTest.jl")
    end
    @testset "Monotone Least Squares" begin
        # include("MLS.jl")
    end
    @testset "2D Banana From Samples" begin
        # include("banana2D.jl")
    end
end
