using MParT
using Distributions

## Create data
dim = 2
N=20_000
N_test = N÷5
data = randn(dim,N)
target = collect(hcat(data[1,:], data[2,:] + data[1,:].^2)')
test = target[:,1:N_test]
train = target[:,N_test+1:end]

## Create objective and map
obj = GaussianKLObjective(train, test)
map_options = MapOptions()
max_order = 2
map = CreateTriangular(dim,dim,max_order,map_options)

## Train the map
train_options = TrainOptions(opt_alg = "LD_SLSQP")
TrainMap(map, obj, train_options)

## Check the testing and training error
@test TestError(obj, map) < 5.
@test TrainError(obj, map) < 5.

## Evaluate test samples after training
pullback_evals = Evaluate(map,test)

## Perform Kolmogorov-Smirnov test
sorted_samples = sort(pullback_evals[:])
dist = Normal()
samps_cdf = cdf.((dist,),sorted_samples)
samps_ecdf = (1:2N_test)/(2N_test)
KS_stat = maximum(abs.(samps_cdf - samps_ecdf))

##
@test KS_stat < 0.1