mset = CreateTotalOrder(1,3)
opts = MapOptions()


function test_CreateComponent_basisType()
    for bType in ["HermiteFunctions", "PhysicistHermite", "ProbabilistHermite"]
        opts = MapOptions(basisType = bType)
        component = CreateComponent(Fix(mset), opts)
        @test numCoeffs(component) == 4
        @test all(CoeffMap(component) .== 0.)
    end
end

function test_CreateComponent_posFuncType()
    for pfType in ["Exp", "SoftPlus"]
        opts = MapOptions(posFuncType = pfType)
        component = CreateComponent(Fix(mset), opts)
        @test numCoeffs(component) == 4
        @test all(CoeffMap(component) .== 0.)
    end
end

function test_CreateComponent_quadTypes()
    for qType in ["AdaptiveSimpson", "ClenshawCurtis", "AdaptiveClenshawCurtis"]
    # AdaptiveSimpson
        opts = MapOptions(quadType = qType)
        component = CreateComponent(Fix(mset), opts)
        @test numCoeffs(component) == 4
        @test all(CoeffMap(component) .== 0.)
    end
end

function test_CreateTriangular()
    triangular = CreateTriangular(2,2,2,opts)
    @test numCoeffs(triangular) == 2 + 7
end

function test_CreateSingleEntryMap_1()
    dim = 7
    activeInd = 1

    mset_csemap = CreateTotalOrder(activeInd, 3)
    component = CreateComponent(Fix(mset_csemap), opts)

    single_entry_map = CreateSingleEntryMap(dim, activeInd, component)
    @test numCoeffs(single_entry_map) == numCoeffs(component)
end

function test_CreateSingleEntryMap_2()
    dim = 7
    activeInd = 7

    mset_csemap = CreateTotalOrder(activeInd, 3)
    component = CreateComponent(Fix(mset_csemap), opts)

    single_entry_map = CreateSingleEntryMap(dim, activeInd, component)
    @test numCoeffs(single_entry_map) == numCoeffs(component)
end

function test_CreateSingleEntryMap_3()
    dim = 7
    activeInd = 4

    mset_csemap = CreateTotalOrder(activeInd, 3)
    component = CreateComponent(Fix(mset_csemap), opts)

    single_entry_map = CreateSingleEntryMap(dim, activeInd, component)
    @test numCoeffs(single_entry_map) == numCoeffs(component)
end

function test_CreateSigmoidMaps()
    input_dim = 6
    num_sigmoid = 5
    centers_len = 2+(num_sigmoid*(num_sigmoid+1)) ÷ 2
    max_degree = 3
    centers = zeros(centers_len)
    center_idx = 1
    bound = 3.
    # Edge terms
    centers[1] = -bound
    centers[2] =  bound
    centers[3] = 0.
    # Sigmoid terms
    for order in 4:num_sigmoid
        for j in 0:order-1
            centers[center_idx] = 1.9*bound*(j-(order-1)/2)/(order-1)
            center_idx += 1
        end
    end
    opts = MapOptions(basisType="HermiteFunctions")
    sig = CreateSigmoidComponent(input_dim, max_degree, centers, opts)
    expected_num_coeffs = binomial(input_dim+max_degree, input_dim)
    @test numCoeffs(sig) == expected_num_coeffs
    mset = FixedMultiIndexSet(input_dim, max_degree)
    sig_mset = CreateSigmoidComponent(mset, centers, opts)
    @test numCoeffs(sig_mset) == size(mset)
    output_dim = input_dim
    centers_total = reduce(hcat, centers for _ in 1:output_dim)
    sig_trimap = CreateSigmoidTriangular(input_dim, output_dim, max_degree, centers_total, opts)
    expected_num_coeffs = sum(binomial(d+max_degree, d) for d in 1:input_dim)
    @test numCoeffs(sig_trimap) == expected_num_coeffs
end

test_CreateComponent_basisType()
test_CreateComponent_posFuncType()
test_CreateComponent_quadTypes()
test_CreateTriangular()
# test_CreateSingleEntryMap_1()
# test_CreateSingleEntryMap_2()
# test_CreateSingleEntryMap_3()
test_CreateSigmoidMaps()