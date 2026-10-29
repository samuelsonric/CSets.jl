module CSets

using Base: oneto
using CliqueTrees: CliqueTrees, BipartiteGraph, cliquetree, parentindex, pointers, residual, targets
using FixedSizeArrays

const FArray{T} = FixedSizeArrayDefault{T}
const FMatrix{T} = FArray{T, 2}
const FVector{T} = FArray{T, 1}
const FScalar{T} = FArray{T, 0}

function two(::Type{I}) where {I}
    return one(I) + one(I)
end

function ispositive(x)
    return x > zero(x)
end

include("span.jl")
include("semiring.jl")
include("partial_map.jl")
include("instance.jl")
include("contraction.jl")
include("decomposition.jl")
include("color.jl")
include("eager.jl")
include("lazy.jl")

greet() = print("Hello World!")

end # module CSets
