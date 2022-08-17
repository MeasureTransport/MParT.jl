using MParT
using Distributions, LinearAlgebra, Statistics
using Optimization, OptimizationOptimJL

##
num_points = 1000
z = randn(2,num_points)
x1 = z[1,:]
x2 = z[2,:] + z[1,:].^2
x = collect([x1 x2]')

test_n_pts = 10_000
test_z = randn(2,test_n_pts)
test_x1 = test_z[1,:]
test_x2 = test_z[2,:] + test_z[1,:].^2
test_x = collect([test_x1 test_x2]')

# For computing reference density
reference_density = MvNormal(I(2))

## Set up map and initialize coefficients
opts = MapOptions()
tri_map = CreateTriangular(2,2,2,opts)
coeffs = zeros(numCoeffs(tri_map))

function obj(coeffs, p)
    tri_map, x, reference_density = p
    SetCoeffs(tri_map, coeffs)
    map_of_x = Evaluate(tri_map, x)
    ref_density_of_map_of_x = logpdf(reference_density, map_of_x)
    log_det = LogDeterminant(tri_map, x)
    -sum(ref_density_of_map_of_x + log_det)/num_points
end

function grad_obj(g, coeffs, p)
    tri_map, x = p
    SetCoeffs(tri_map, coeffs)
    map_of_x = Evaluate(tri_map, x)
    grad_ref_density_of_map_of_x = -CoeffGrad(tri_map, x, map_of_x)
    grad_log_det = LogDeterminantCoeffGrad(tri_map, x)
    g .= -sum(grad_ref_density_of_map_of_x + grad_log_det, dims=2)/num_points
end

## Plot before Optimization
u0 = CoeffMap(tri_map)
p = (tri_map, x, reference_density)
fcn = OptimizationFunction(obj, grad = grad_obj)
prob = OptimizationProblem(fcn, u0, p, g_tol = 1e-16)

## Optimize

sol = solve(prob, BFGS())

u_final = sol.u
SetCoeffs(tri_map, u_final)
map_of_test_x = Evaluate(tri_map, test_x)

mean_of_map = mean(map_of_test_x, dims=2)
cov_of_map = cov(map_of_test_x, dims=2)

## Test (heuristics)
tol = 0.5
@test norm(mean_of_map) < tol
@test norm(cov_of_map - I(2)) < tol