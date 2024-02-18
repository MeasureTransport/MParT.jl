# MultiIndex test

multis = [0 1; 2 0]
msetFromArray = MultiIndexSet(multis)

dim = 3
power = 4
msetTotalOrder = CreateTotalOrder(dim, power)

function test_max_degrees()
    @test Int.(MParT.MaxOrders(msetFromArray)) == [2, 1]
    @test Int.(MParT.MaxOrders(msetTotalOrder)) == [4, 4, 4]
end

function test_reduced_margin()
    msetTotalOrder2 = CreateTotalOrder(dim, power+1)
    msetTotalOrder_rm = MParT.ReducedMargin(msetTotalOrder)
    @test length(msetTotalOrder_rm) == Size(msetTotalOrder2) - Size(msetTotalOrder)
    @test all([sum(midx) == power+1 for midx in msetTotalOrder_rm])
    msetTotalOrder_rm_dim = MParT.ReducedMarginDim(msetTotalOrder,2)
    @test all([sum(midx) == power+1 for midx in msetTotalOrder_rm_dim])
    @test length(msetTotalOrder_rm_dim) < length(msetTotalOrder_rm)

    # Tests weird memory bug where ReducedDim returns objects that get garbage collected
    dims_add = [1, 2]
    # Without the additional constructor, the MultiIndex objects get gc'd
    msetTotalOrder_rm_dim = reduce(vcat, MultiIndex.(MParT.ReducedMarginDim(msetTotalOrder, d)) for d in dims_add)
    mset_strings = string.(string.(msetTotalOrder_rm_dim))
    # Just test that things don't crash, i.e. we didn't segfault
    @test mset_strings[1] isa AbstractString
end

function test_at()
    @test msetFromArray[1] == MultiIndex([0, 1])
    @test msetFromArray[2] == MultiIndex([2, 0])

    @test msetTotalOrder[1] == MultiIndex([0, 0, 0])
    @test msetTotalOrder[2] == MultiIndex([0, 0, 1])
    last_idx = Size(msetTotalOrder)
    @test msetTotalOrder[last_idx] == MultiIndex([4, 0, 0])
end

test_max_degrees()
test_reduced_margin()
test_at()