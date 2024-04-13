# Wrapping code to make the Julia module usable
module MParT

using CxxWrap
using MParT_jll
import Libdl
@wrapmodule ()->libmpartjl :MParT_julia_module
import Base: getindex, lastindex, show, iterate, vec

ConditionalMapBasePtr = CxxWrap.StdLib.SharedPtr{<:ConditionalMapBase}
for op = (:Evaluate, :Gradient, :Inverse, :inputDim, :outputDim,
            :LogDeterminant, :LogDeterminantCoeffGrad, :LogDeterminantInputGrad,
            :numCoeffs, :CoeffMap, :CoeffGrad, :SetCoeffs, :TestError)
    eval(quote
        $op(obj::CxxWrap.StdLib.SharedPtr, args...) = $op(obj[], args...)
    end)
end

function __init__()
    @initcxx
    threads = get(ENV, "KOKKOS_NUM_THREADS", nothing)
    opts = StdVector{StdString}()
    if !isnothing(threads)
        push!(opts, StdString("kokkos_num_threads"))
        push!(opts, StdString(string(threads)))
    end
    length(opts) > 0 && @info "Using MParT options: "*join(string.(opts),", ")
    Initialize(opts)
end

"""
    Concurrency()
See how many threads MParT is using.
"""
Concurrency

function Base.iterate(mset::MultiIndexSet)
    Size(mset) < 1 && return nothing
    mset[1],2
end

function Base.iterate(mset::MultiIndexSet,state::Int)
    state > Size(mset) && return nothing
    return mset[state], state+1
end

"""
    Size(mset::MultiIndexSet)
Number of MultiIndex objects in a MultiIndexSet `mset`.
"""
Size

"""
    MultiIndexSet(A::AbstractVecOrMat{<:Integer})

Create a set of MultiIndices from the rows of `A`.

These indices represent a basis for a multivariate expansion or,
further, monotone expansion. Each element of the set is a MultiIndex
representing one basis function via the degrees in each dimension.

# Example
```jldoctest
julia> # Functions like: c_1xy^2z^3 + c_2xyz + c_3

julia> A = [1 2 3;1 1 1; 0 0 0];

julia> mset = MultiIndexSet(A);
```
See also [`MultiIndex`](@ref), [`FixedMultiIndexSet`](@ref), [`Fix`](@ref)
"""
MultiIndexSet

"""
    FixedMultiIndexSet(dim::Int, p::Int)
Creates a FixedMultiIndexSet with dimension `dim` and total order `p`.

A FixedMultiIndexSet is just a compressed, efficient way of representing a MultiIndexSet, but without as many bells and whistles.

See also: [`MultiIndex`](@ref), [`MultiIndexSet`](@ref)
"""
FixedMultiIndexSet

"""
    CreateTotalOrder(dim::Int, p::Int)
Creates a total order `p` MultiIndexSet object in dimension `dim`.

See also: [`MultiIndexSet`](@ref)
"""
CreateTotalOrder

"""
    Fix(mset::MultiIndexSet, compress::Bool = true)
Take `mset` and turn it into a `FixedMultiIndexSet` that can be `compress`ed.

See also [`MultiIndex`](@ref), [`MultiIndexSet`](@ref), [`FixedMultiIndexSet`](@ref)
"""
Fix(mset::MultiIndexSet) = Fix(mset, true)

MultiIndexSet(A::AbstractMatrix{<:Integer}) = MultiIndexSet(Cint.(collect(A)))
MultiIndexSet(A::AbstractVector{<:Integer}) = MultiIndexSet(Cint.(collect(reshape(A, length(A), 1))))
Base.getindex(A::MultiIndex, i::AbstractVector{<:Integer}) = getindex.((A,), i)
Base.lastindex(A::MultiIndex) = length(A)
Base.vec(A::MultiIndex) = Int.(ToVector(A))

# Not implemented yet
# Base.getindex(A::CxxWrap.reference_type_union(MParT.TriangularMap), s::Base.UnitRange) = Slice(A, first(s), last(s))

"""
    Evaluate(map, points)
Evaluates the function `map` at `points`, where each column of `points` is a different sample.

If `map` ``:\\mathbb{R}^m\\to\\mathbb{R}^n``, then `points` ``\\in\\mathbb{R}^m\\times\\mathbb{R}^k``, where ``k`` is the number of points.
"""
Evaluate

"""
    Inverse(map, y, x)
If `map` represents function ``T(y,x)``, then this function calculates ``T(y,\\cdot)^{-1}(x)``.

If `map` is square, **you still require `y`**, but it can be a 0 x k matrix.
"""
Inverse

"""
    MapOptions(;kwargs...)
Creates options for parameterized map.

All possible keyword arguments are in example, with some important arguments described below. See C++ documentation for an exhaustive description.

# Arguments
- `basisType::String`: Includes "ProbabilistHermite", "PhysicistHermite", "HermiteFunctions"
- `basisLB::Float64`,`basisUB::Float64`: The bounds for where we start linearizing the map. These default to infinities, but often making the data near the origin and setting them to a small finite number (e.g. +-3) works well.

# Example
```jldoctest
julia> MapOptions(basisType="HermiteFunctions", basisLB=-3., basisUB=3.)
basisType = HermiteFunctions
basisLB = -3
basisUB = 3
basisNorm = true
posFuncType = SoftPlus
quadType = AdaptiveSimpson
quadAbsTol = 1e-06
quadRelTol = 1e-06
quadMaxSub = 30
quadMinSub = 0
quadPts = 5
contDeriv = true
nugget = 0

```
See also [`CreateComponent`](@ref), [`TriangularMap`](@ref), [`CreateTriangular`](@ref)
"""
function MapOptions(;kwargs...)
    opts = __MapOptions()
    for kwarg in kwargs
        field = Symbol("__"*string(first(kwarg))*"!")
        value = last(kwarg)
        if value isa String
            value = MParT.eval(Meta.parse("__"*value))
        end
        getfield(MParT, field)(opts, value)
    end
    opts
end

"""
    `TrainOptions(;kwargs...)`
Creates options for using `TrainMap` to train a transport map.

See example for possible arguments.

# Examples
```jldoctest
julia> TrainOptions(opt_alg="LD_SLSQP", opt_maxeval = 1_000_000)
opt_alg = LD_SLSQP
opt_stopval = -inf
opt_ftol_rel = 0.001
opt_ftol_abs = 0.001
opt_xtol_rel = 0.0001
opt_xtol_abs = 0.0001
opt_maxeval = 1000000
opt_maxtime = inf
verbose = 0

```
See also [`TrainMap`](@ref), [`CreateGaussianKLObjective`](@ref)
"""
function TrainOptions(;kwargs...)
    opts = __TrainOptions()
    for kwarg in kwargs
        field = Symbol("__"*string(first(kwarg))*"!")
        value = last(kwarg)
        getfield(MParT, field)(opts, value)
    end
    opts
end

"""
    CreateGaussianKLObjective(train)
    CreateGaussianKLObjective(train, outputDim)
    CreateGaussianKLObjective(train, test)
    CreateGaussianKLObjective(train, test, outputDim)

Create an objective for the optimization problem of training a
transport map.

Currently only supports variational simulation-based inference, i.e.
creating a map that uses samples `train` from target distribution and a
Gaussian reference distribution of dimension `outputDim`. If `outputDim`
is zero, this is equivalent to when `outputDim == size(train,1)`.

# Arguments
- `train::Matrix{Float64}`: mandatory training dataset
- `test::Matrix{Float64}`: optional test dataset
- `outputDim::Int = 0`: Dimensions of output of map

# Examples
```jldoctest
julia> using Random; rng = Random.Xoshiro(1);

julia> # Replace RNG type with MersenneTwister for <1.7

julia> N, inDim, outDim = 1000, 4, 2;

julia> samples = 2*randn(rng, inDim, N) .+ 5;

julia> obj1 = CreateGaussianKLObjective(samples);

julia> obj2 = CreateGaussianKLObjective(samples, outDim);

julia> train, test = samples[:,1:500], samples[:,501:end];

julia> obj3 = CreateGaussianKLObjective(train, test);

julia> obj4 = CreateGaussianKLObjective(train, test, outDim);
```
See also [`TrainMap`](@ref), [`TrainOptions`](@ref)
"""
CreateGaussianKLObjective

CreateGaussianKLObjective(train::Matrix{Float64}) = CreateGaussianKLObjective(train,0)
CreateGaussianKLObjective(train::Matrix{Float64},test::Matrix{Float64}) = CreateGaussianKLObjective(train,test,0)


"""
    TestError(obj::MapObjective, map)
Uses the test dataset in obj to evaluate the error of `map`.
"""
TestError

"""
    `ATMOptions(;kwargs...)`
Options for using the Adaptive Transport Map algorithm from [Baptista, et al.](https://arxiv.org/pdf/2009.10303.pdf)

Inherits all keywords from `MapOptions` and `TrainOptions`, plus the arguments below.

# Arguments
- `maxPatience::Int`: Number of "stationary" algorithm iterations tolerated
- `maxSize::Int`: the _total_ number of multiindices in the _entire map_ that the algorithm is allowed to add. Should be larger than the sum of the sizes of all multiindex sets across all dimensions for the map
- `maxDegrees::MultiIndex`: The maximum degree of any expansion term for each dimension (should be length of dimensions of the map)

# Examples
```jldoctest
julia> maxDegrees = MultiIndex(2,3); # limit both dimensions by order 3

julia> ATMOptions(maxDegrees=maxDegrees);
```
See also [`TrainMapAdaptive`](@ref), [`TrainOptions`](@ref), [`MapOptions`](@ref)
"""
function ATMOptions(;kwargs...)
    opts = __ATMOptions()
    for kwarg in kwargs
        field = Symbol("__"*string(first(kwarg))*"!")
        value = last(kwarg)
        if value isa String && !startswith(value, "opt") && value != "verbose"
            value = MParT.eval(Meta.parse("__"*value))
        end
        getfield(MParT, field)(opts, value)
    end
    opts
end

# To print MapOptions, TrainOptions objects
Base.show(io::IO,::MIME"text/plain", opts::__MapOptions) = print(io,string(opts))
Base.show(io::IO,::MIME"text/plain", opts::__TrainOptions) = print(io,string(opts))
Base.show(io::IO,::MIME"text/plain",opts::__ATMOptions) = print(io, string(opts))


"""
    DeserializeMap(filename::String)
REQUIRES CEREAL INSTALLATION. Deserializes a map and returns its input dimension, output dimension, and coefficient.
"""
function DeserializeMap(filename::String)
    dims = Cint[0,0]
    coeffs = __DeserializeMap(filename, dims);
    dims[1], dims[2], coeffs
end

"""
    Deserialize(obj, filename)
Deserializes `filename` and puts the contents in `obj`. REQUIRES CEREAL INSTALLATION.

The object `obj` can be of type `MapOptions` or `FixedMultiIndexSet`. This will create a new pointer-- other objects with the same pointer will not be modified, but the contents of `obj` will now point to the deserialized object.
"""
Deserialize

"""
    Serialize(obj, filename)
Serializes `obj` into file `filename`. REQUIRES CEREAL INSTALLATION.
"""
Serialize

"""
    TrainMap(map, obj::MapObjective, opts::TrainOptions)
Trains `map` according to the objective `obj` with training options `opts`.
"""
TrainMap

"""
    TrainMapAdaptive(msets, objective, options)
Implements the ATM algorithm [Baptista, et al.](https://arxiv.org/pdf/2009.10303.pdf)

Takes in initial guess of multiindex sets for each output dimension and adapts
those sets to better approximate the probability distribution of interest using monotone transport maps.

# Examples
"""
function TrainMapAdaptive(msets::Vector{<:MultiIndexSet},obj::CxxWrap.StdLib.SharedPtr{<:MapObjective}, opts::__ATMOptions)
    msets_vec = [CxxRef(mset) for mset in msets]
    TrainMapAdaptive(msets_vec, obj, opts)
end


"""
    TriangularMap(maps::Vector, move_coeffs::Bool = true)
Creates a `TriangularMap` from a vector of `ConditionalMapBase` objects.

TODO: The new object takes ownership of the coeffs of the maps in `maps` if
`move_coeffs` is true.

# Examples
```jldoctest
julia> dim, order = 5, 3;

julia> msets = [FixedMultiIndexSet(d, order) for d in 1:dim];

julia> opts = MapOptions();

julia> components = [CreateComponent(mset, opts) for mset in msets];

julia> trimap = TriangularMap(components);
```
"""
function TriangularMap(maps::Vector, move_coeffs::Bool = true)
    maps = StdVector([map for map in maps])
    TriangularMap(maps, move_coeffs)
end

"""
    CreateTriangular(inDim::Int, outDim::Int, p::Int, opts::MapOptions)
Creates a total order `p` map with dimensions `inDim` and `outDim` with specifications `opts`.
"""
CreateTriangular

"""
    CreateComponent(mset::FixedMultiIndexSet, opts::MapOptions)
Create a single-output component with approximation order given by `mset` and specified by `opts`
"""
CreateComponent

"""
    ComposedMap(maps::Vector)
Creates a `ComposedMap` from a vector of `ConditionalMapBase` objects.
""" # TODO: Example
function ComposedMap(maps::Vector{<:ConditionalMapBasePtr})
    maps_std = StdVector([map for map in maps])
    ComposedMap(maps_std)
end

"""
    MultiIndex(A::AbstractVector{<:Int})
MultiIndex defines the order of one term of a function expansion for each dimensions.

# Example
```jldoctest
julia> degrees = [3,2,1]; # Represents polynomial basis function similar to x^3y^2z^1

julia> midx = MultiIndex(degrees);

julia> midx[3]
0x00000001
```
See also [`MultiIndexSet`](@ref), [`FixedMultiIndexSet`](@ref), [`Fix`](@ref)
"""
MultiIndex(A::AbstractVector{<:Int}) = MultiIndex(StdVector(Cuint.(A)))

# MultiIndex-related exports
export MultiIndex, MultiIndexSet, FixedMultiIndexSet
export Fix, CreateTotalOrder, Size, addto!
# ParameterizedFunctionBase-related exports
export CoeffMap, SetCoeffs, numCoeffs, inputDim, outputDim
export Evaluate, CoeffGrad, Gradient
# ConditionalMapBase-related exports
export ConditionalMapBasePtr, GetBaseFunction, LogDeterminant, LogDeterminantCoeffGrad, Inverse
# TriangularMap-related exports
export TriangularMap, InverseInplace, GetComponent
# AffineMap-related exports
export AffineMap, AffineFunction
# ComposedMap-related exports
export ComposedMap
# MapFactory-related exports
export CreateComponent, CreateTriangular, CreateSigmoidComponent, CreateSigmoidTriangular
# MapOptions-related exports
export MapOptions
# Serialization-related exports
export Serialize, Deserialize, DeserializeMap
# Map training related exports
export CreateGaussianKLObjective, TrainOptions, TrainMap, TestError
# ATM-related exports
export TrainMapAdaptive, ATMOptions
# Other important utils
export Concurrency
end
