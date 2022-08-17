using MParT
using Distributions, LinearAlgebra, Statistics, Optimization, OptimizationOptimJL

##

# Geometry
num_points = 1000
xmin, xmax = 0,4
x = collect(range(xmin, xmax, length=num_points)')

# Measurements
noisesd = 0.4

# Notes: data might not be monotone bc of the noise
# but we assume the true underlying function is monotone
y_true = 2*(x .> 2)
y_noise = noisesd*randn(1,num_points)
y_measured = y_true + y_noise

# Create MultiIndexSet
multis = collect(reshape(0:5,6,1))
mset = MultiIndexSet(multis)
fixed_mset = Fix(mset, true)

# Set MapOptions and make map
opts = MapOptions()
monotoneMap = CreateComponent(fixed_mset, opts)

##
# Least Squares objective
function objective(coeffs,p)
    monotoneMap, x, y_measured = p
    SetCoeffs(monotoneMap, coeffs)
    map_of_x = Evaluate(monotoneMap, x)
    norm(map_of_x - y_measured)^2/size(x,2)
end

# Before Optimization
map_of_x_before = Evaluate(monotoneMap, x)
u0 = CoeffMap(monotoneMap)
p = (monotoneMap, x, y_measured)
fcn = OptimizationFunction(objective)
prob = OptimizationProblem(fcn, u0, p)

# Optimize
sol = solve(prob, NelderMead())
u_final = sol.u
SetCoeffs(monotoneMap, u_final)

## After Optimization
map_of_x_after = Evaluate(monotoneMap, x)
error_after = objective(u_final, p)
@test error_after ≈ 0. atol=1e-1
