# Wrapping code to make the Julia module usable
module MParT
    using CxxWrap
    using MParT_jll
    import Libdl
    @wrapmodule "libmpartjl" :MParT_julia_module

    function __init__()
        @initcxx
        threads = get(ENV, "KOKKOS_NUM_THREADS", nothing)
        opts = isnothing(threads) ? [] : ["kokkos_num_threads", threads]
        length(opts) > 0 && @info "Using MParT options: "*string(string.(opts))
        Initialize(StdVector(StdString.(opts)))
    end

    module BasisTypes
        using CxxWrap
        @wrapmodule("libmpartjl", :BasisType_julia_module)
        function __init__()
            @initcxx
        end
    end

    module PosFuncTypes
        using CxxWrap
        @wrapmodule("libmpartjl", :PosFuncType_julia_module)
        function __init__()
            @initcxx
        end
    end

    module QuadTypes
        using CxxWrap
        @wrapmodule("libmpartjl", :QuadType_julia_module)
        function __init__()
            @initcxx
        end
    end

    MultiIndexSet(A::AbstractMatrix{<:Integer}) = MultiIndexSet(Cint.(collect(A)))
    MultiIndexSet(A::AbstractVector{<:Integer}) = MultiIndexSet(Cint.(collect(reshape(A, length(A), 1))))

    function MapOptions(;kwargs...)
        opts = __MapOptions()
        for kwarg in kwargs
            str = "__"*string(first(kwarg))*"!"
            getfield(MParT, Symbol(str))(opts, last(kwarg))
        end
        opts
    end

    export SetCoeffs, MapOptions, MultiIndexSet, FixedMultiIndexSet,
           Fix, CoeffMap, LogDeterminant, CreateTotalOrder,
           Evaluate, numCoeffs, CoeffGrad, Gradient,
           LogDeterminantCoeffGrad, Inverse,
           CreateComponent, CreateTriangular,
           BasisTypes, PosFuncTypes, QuadTypes
end