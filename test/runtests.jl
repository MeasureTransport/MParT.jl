using Test

@testset "MParT.jl" verbose=true begin
    @testset verbose=true "Affine, Composed Map" begin
        include("mapTypeTest.jl")
    end
    @testset verbose=true "Monotone Least Squares" begin
        include("MLS.jl")
    end
    @testset verbose=true "2D Banana From Samples" begin
        include("banana2D.jl")
    end
end
