# Wrapping code to make the Julia module usable
module MParT
    using CxxWrap
    using MParT_jll
    import Libdl
    @wrapmodule libmpartjl :MParT_julia_module
    import Base: getindex, lastindex, show

    mapSubtypeAlias{T} = Union{T,<:CxxWrap.CxxWrapCore.SmartPointer{T}}

    function __init__()
        @initcxx
        threads = get(ENV, "KOKKOS_NUM_THREADS", nothing)
        opts = isnothing(threads) ? [] : ["kokkos_num_threads", threads]
        length(opts) > 0 && @info "Using MParT options: "*string(string.(opts))
        Initialize(StdVector(StdString.(opts)))
    end

    # Provides shortcuts for MultiIndexSet for Julia-style arrays
    MultiIndexSet(A::AbstractMatrix{<:Integer}) = MultiIndexSet(Cint.(collect(A)))
    MultiIndexSet(A::AbstractVector{<:Integer}) = MultiIndexSet(Cint.(collect(reshape(A, length(A), 1))))
    Base.getindex(A::MultiIndex, i::AbstractVector{<:Integer}) = getindex.((A,), i)
    Base.lastindex(A::MultiIndex) = length(A)

    Base.getindex(A::CxxWrap.reference_type_union(MParT.TriangularMap), s::Base.UnitRange) = Slice(A, first(s), last(s))

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

    # To print MapOptions, TrainOptions objects
    Base.show(io::IO,::MIME"text/plain", opts::__MapOptions) = print(io,string(opts))
    Base.show(io::IO,::MIME"text/plain", opts::__TrainOptions) = print(io,string(opts))


    function DeserializeMap(filename::String)
        dims = Cint[0,0]
        coeffs = __DeserializeMap(filename, dims);
        dims[1], dims[2], coeffs
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

    """
        `TriangularMap(maps::Vector)`
    Creates a `TriangularMap` from a vector of `ConditionalMapBase` objects
    """
    function TriangularMap(maps::Vector{<:CxxWrap.StdLib.SharedPtr{<:ConditionalMapBase}})
        maps_cmb = Vector{CxxWrap.StdLib.SharedPtr{ConditionalMapBase}}(undef, length(maps))
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
    function ComposedMap(maps::Vector{<:CxxWrap.StdLib.SharedPtr{<:ConditionalMapBase}})
        maps_cmb = Vector{CxxWrap.StdLib.SharedPtr{ConditionalMapBase}}(undef, length(maps))
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
    export GetBaseFunction, LogDeterminant, LogDeterminantCoeffGrad, Inverse
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
    export GaussianKLObjective, TrainOptions, TrainMap, TestError
    # Other important utils
    export Concurrency
end
