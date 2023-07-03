using MParT
using Distributions, LinearAlgebra, Statistics, CxxWrap

A = Float64[1 2; 4 5]
b = Float64[5, 6]

N_samples = 1000

a_map1 = AffineMap(A, b)
a_map2 = AffineMap(A)
a_map3 = AffineMap(b)
a_fun = AffineFunction(A, b)
c_map = ComposedMap(StdVector(MParT.ConditionalMapBase[a_map2, a_map3]))

## Test the eval
x = randn(size(A, 2), N_samples)
y_true = A * x .+ b
y1 = Evaluate(a_map1, x)
y2 = Evaluate(a_map2, x)
y3 = Evaluate(a_map3, y2)
y_fun = Evaluate(a_fun, x)
y_c = Evaluate(c_map, x)
@test y1 ≈ y_true
@test y3 ≈ y_true
@test y_c ≈ y_true
@test y_fun ≈ y_true

## Test the gradient
sens = randn(size(A, 1), N_samples)
g1 = Gradient(a_map1, x, sens)
g2 = Gradient(a_map2, x, sens)
g3 = Gradient(a_map3, x, sens)
g_fun = Gradient(a_fun, x, sens)
g_c = Gradient(c_map, x, sens)

@test g1 ≈ A'sens
@test g2 ≈ A'sens
@test g3 ≈ sens
@test g_c ≈ A'sens
@test g_fun ≈ A'sens

## Test the log determinant
true_ld = log(abs(det(A[:, end-size(A, 1)+1:end])))
ld1 = LogDeterminant(a_map1, x)
ld2 = LogDeterminant(a_map2, x)
ld3 = LogDeterminant(a_map3, x)
ld_c = LogDeterminant(c_map, x)

@test ld1 ≈ fill(true_ld, size(x,2))
@test ld2 ≈ fill(true_ld, size(x,2))
@test ld3 ≈ zeros(size(x,2))
@test ld_c ≈ fill(true_ld, size(x,2))

## Rectangular A
A = Float64[1 2 3; 4 5 6]
b = Float64[5, 6]
a_rect1 = AffineMap(A, b)
a_rect2 = AffineMap(A)
a_funr = AffineFunction(A, b)

## Test the eval
x = randn(size(A, 2), N_samples)
y_true = A * x .+ b
y1 = Evaluate(a_rect1, x)
y2 = Evaluate(a_rect2, x)
y3 = Evaluate(a_map3, y2)
y_fun = Evaluate(a_funr, x)
@test y1 ≈ y_true
@test y3 ≈ y_true
@test y_fun ≈ y_true

## Test the gradient
sens = randn(size(A, 1), N_samples)
g1 = Gradient(a_rect1, x, sens)
g2 = Gradient(a_rect2, x, sens)
g_fun = Gradient(a_funr, x, sens)

@test g1 ≈ A'sens
@test g2 ≈ A'sens
@test g_fun ≈ A'sens

## Test the log determinant
true_ld = log(abs(det(A[:, end-size(A, 1)+1:end])))
ld1 = LogDeterminant(a_rect1, x)
ld2 = LogDeterminant(a_rect2, x)
ld3 = LogDeterminant(a_map3, x)

@test ld1 ≈ fill(true_ld, size(x,2))
@test ld2 ≈ fill(true_ld, size(x,2))
@test ld3 ≈ zeros(size(x,2))