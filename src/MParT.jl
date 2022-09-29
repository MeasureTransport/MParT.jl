# Wrapping code to make the Julia module usable
module MParT
    using CxxWrap
    using MParT_jll
    import Libdl
    @wrapmodule libmpartjl :MParT_julia_module
    import Base: getindex, lastindex

    mapSubtypeAlias{T} = Union{T,CxxWrap.CxxWrapCore.SmartPointer{T}}

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
            value = MParT.eval(Meta.parse("__"*last(kwarg)))
            getfield(MParT, field)(opts, value)
        end
        opts
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
    export InverseInplace, GetComponent
    # AffineMap-related exports
    export AffineMap, AffineFunction
    # MapFactory-related exports
    export CreateComponent, CreateTriangular
    # MapOptions-related exports
    export MapOptions
    # Other important utils
    export mapSubtypeAlias
end
