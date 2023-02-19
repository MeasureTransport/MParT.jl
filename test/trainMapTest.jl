using Distributions

function KS_statistic(map, samples)
    pullback_evals = Evaluate(map,samples)

    ## Perform Kolmogorov-Smirnov test
    sorted_samples = sort(pullback_evals[:])
    dist = Normal()
    samps_cdf = cdf.((dist,),sorted_samples)
    samps_ecdf = (1:length(sorted_samples))/length(sorted_samples)
    maximum(abs.(samps_cdf - samps_ecdf))
end

## Create data
dim = 2
N=20_000
N_test = N÷5
data = randn(dim+1,N)
target = collect(hcat(data[1,:],data[2,:], data[3,:] + data[2,:].^2)')
test = target[:,1:N_test]
train = target[:,N_test+1:end]

## Create objective and map
obj1 = CreateGaussianKLObjective(train,test,1)
obj2 = CreateGaussianKLObjective(train,test,2)
obj3 = CreateGaussianKLObjective(train,test)
map_options = MapOptions()
max_order = 2

map1 = CreateComponent(FixedMultiIndexSet(dim+1,max_order),map_options)
map2 = CreateTriangular(dim+1,dim,max_order,map_options)
map3 = CreateTriangular(dim+1,dim+1,max_order,map_options)

## Train the map
train_options = TrainOptions()
TrainMap(map1, obj1, train_options)
TrainMap(map2, obj2, train_options)
TrainMap(map3, obj3, train_options)

## Check the testing and training error
@test TestError(obj1, map1) < 5.
@test TestError(obj2, map2) < 5.
@test TestError(obj3, map3) < 5.

## Evaluate test samples after training
KS_stat1 = KS_statistic(map1,test)
KS_stat2 = KS_statistic(map2,test)
KS_stat3 = KS_statistic(map3,test)

##
@test KS_stat1 < 0.1
@test KS_stat2 < 0.1
@test KS_stat3 < 0.1