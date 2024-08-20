# Test methods of ConditionalMapBase object
opts = MapOptions()

multis = [0;;1]'# linear
mset = Fix(MultiIndexSet(multis), true)

component = CreateComponent(mset, opts)
num_samples = 100
x = randn(1,num_samples)



function test_numCoeffs()
    @test numCoeffs(component) == 2
end

function test_CoeffsMap()
    SetCoeffs(component, zeros(numCoeffs(component)))
    @test all(CoeffMap(component) .== [0,0])

    coeffs = randn(numCoeffs(component))
    SetCoeffs(component, coeffs)
    @test all(CoeffMap(component) .== coeffs)
end


function test_CoeffBounds()
    lb, ub = CoeffBounds(component)
    @test size(lb,1) == numCoeffs(component)
    @test size(ub,1) == numCoeffs(component)

    @test maximum(lb) == -Inf
    @test minimum(ub) == Inf
end

function test_Evaluate()
    @test size(Evaluate(component, x)) == (1,num_samples)
end

function test_LogDeterminant()
    @test size(LogDeterminant(component, x)) == (num_samples,)
end

function test_Inverse()
    coeffs = randn(numCoeffs(component))
    SetCoeffs(component, coeffs)
    y = Evaluate(component, x)
    x_ = Inverse(component, zeros(1,num_samples), y)
    # @info "" size(x_) size(x)
    @test isapprox(x_, x, atol=1e-3)
end

test_numCoeffs()
test_CoeffsMap()
test_CoeffBounds()
test_Evaluate()
test_LogDeterminant()
test_Inverse()