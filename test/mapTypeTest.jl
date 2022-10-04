using MParT
using Distributions, LinearAlgebra, Statistics

A = Float64[1 2; 3 4]
b = Float64[5, 6]

a_map1 = AffineMap(A, b)
a_map2 = AffineMap(A)
a_map3 = AffineMap(b)
c_map  = ComposedMap([a_map2, a_map3])

## Test the eval
x = randn(2,1000)
y_true = A*x .+ b
y1 = Evaluate(a_map1, x)
y2 = Evaluate(a_map2, x)
y3 = Evaluate(a_map3, x)
y_c = Evaluate(c_map, x)
@test y1 ≈ y_true
@test y2+y3 ≈ y_true
@test y_c ≈ y_true

## Test the gradient
sens = randn(2,1000)
g1 = Gradient(a_map1, x, sens)
g2 = Gradient(a_map2, x, sens)
g3 = Gradient(a_map3, x, sens)
g_c = Gradient(c_map, x, sens)

@test g1 ≈ A'sens
@test g2 ≈ A'sens
@test g3 ≈ sens
@test g_c ≈ A'sens

## Test the log determinant