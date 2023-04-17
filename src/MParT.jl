# Wrapping code to make the Julia module usable
module MParT
    using CxxWrap
    using MParT_jll
    import Libdl
    @wrapmodule libmpartjl :MParT_julia_module
    import Base: getindex, lastindex, show, iterate

    ConditionalMapBasePtr = CxxWrap.StdLib.SharedPtr{<:ConditionalMapBase}

    function __init__()
        @initcxx
        threads = get(ENV, "KOKKOS_NUM_THREADS", nothing)
        opts = isnothing(threads) ? [] : ["kokkos_num_threads", threads]
        length(opts) > 0 && @info "Using MParT options: "*string(string.(opts))
        Initialize(StdVector(StdString.(opts)))
    end

    function Base.iterate(mset::MultiIndexSet)
        Size(mset) < 1 && return nothing
        mset[1],2
    end

    function Base.iterate(mset::MultiIndexSet,state::Int)
        state > Size(mset) && return nothing
        return mset[state], state+1
    end

    # Provides shortcuts for MultiIndexSet for Julia-style arrays
    MultiIndexSet(A::AbstractMatrix{<:Integer}) = MultiIndexSet(Cint.(collect(A)))
    MultiIndexSet(A::AbstractVector{<:Integer}) = MultiIndexSet(Cint.(collect(reshape(A, length(A), 1))))
    Base.getindex(A::MultiIndex, i::AbstractVector{<:Integer}) = getindex.((A,), i)
    Base.lastindex(A::MultiIndex) = length(A)

    # Not implemented yet
    # Base.getindex(A::CxxWrap.reference_type_union(MParT.TriangularMap), s::Base.UnitRange) = Slice(A, first(s), last(s))

    """
        `MapOptions(;kwargs...)`
    Takes the fields from MParT's `MapOptions` as keyword arguments, and
    assigns the field value based on a String from the kwarg value, e.g.
    ```julia
    julia> using MParT

    julia> MapOptions(basisType="HermiteFunctions")
    ```
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
    Takes the fields from MParT's `TrainOptions` as keyword arguments, and
    assigns the field value based on a String from the kwarg value, e.g.
    ```julia
    julia> using MParT

    julia> TrainOptions(opt_alg="LD_SLSQP")
    ```
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

    CreateGaussianKLObjective(train::Matrix{Float64}) = CreateGaussianKLObjective(train,0)
    CreateGaussianKLObjective(train::Matrix{Float64},test::Matrix{Float64}) = CreateGaussianKLObjective(train,test,0)

    """
        `ATMOptions(;kwargs...)`
    Takes the fields from MParT's `ATMOptions` as keyword arguments, and
    assigns the field value based on a String from the kwarg value, e.g.
    ```julia
    julia> using MParT

    julia> maxDegrees = MultiIndex(2,3) # limit both dimensions by order 3

    julia> ATMOptions(opt_alg="LD_SLSQP", maxDegrees=maxDegrees)
    ```
    """
    function ATMOptions(;kwargs...)
        opts = __ATMOptions()
        for kwarg in kwargs
            field = Symbol("__"*string(first(kwarg))*"!")
            value = last(kwarg)
            getfield(MParT, field)(opts, value)
        end
        opts
    end

    # To print MapOptions, TrainOptions objects
    Base.show(io::IO,::MIME"text/plain", opts::__MapOptions) = print(io,string(opts))
    Base.show(io::IO,::MIME"text/plain", opts::__TrainOptions) = print(io,string(opts))


    function DeserializeMap(filename::String)
        dims = Cint[0,0]
        coeffs = __DeserializeMap(filename, dims);
        dims[1], dims[2], coeffs
    end


    function TrainMapAdaptive(msets::Vector{<:MultiIndexSet},obj::CxxWrap.StdLib.SharedPtr{<:MapObjective}, opts::__ATMOptions)
        msets_vec = [CxxRef(mset) for mset in msets]
        TrainMapAdaptive(msets_vec, obj, opts)
    end


    """
        `TriangularMap(maps::Vector)`
    Creates a `TriangularMap` from a vector of `ConditionalMapBase` objects
    """
    function TriangularMap(maps::Vector{<:ConditionalMapBasePtr})
        maps_cmb = Vector{ConditionalMapBasePtr}(undef, length(maps))
        for (i, map) in enumerate(maps)
            maps_cmb[i] = map
        end
        maps_std = StdVector(maps_cmb)
        TriangularMap(maps_std)
    end

    """
        `ComposedMap(maps::Vector)`
    Creates a `ComposedMap` from a vector of `ConditionalMapBase` objects.
    """
    function ComposedMap(maps::Vector{<:ConditionalMapBasePtr})
        maps_cmb = Vector{ConditionalMapBasePtr}(undef, length(maps))
        for (i, map) in enumerate(maps)
            maps_cmb[i] = map
        end
        maps_std = StdVector(maps_cmb)
        ComposedMap(maps_std)
    end

    # MultiIndex-related exports
    export MultiIndex, MultiIndexSet, FixedMultiIndexSet
    export Fix, CreateTotalOrder, Size
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
    export CreateComponent, CreateTriangular
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
