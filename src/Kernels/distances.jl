## Distances
""" 
    Distance

    A struct of abstract type Distance produces the distance between two `global` descriptors, or features. Not all distances might be compatible with all types of features.
"""
abstract type Distance end

"""
    Forstner <: Distance 
        α :: Regularization parameter

    Computes the squared Forstner distance between two positive semi-definite matrices.
"""
struct Forstner <: Distance
    α::Any
end
function Forstner(; α = 1e-6)
    Forstner(α)
end

function compute_distance(
    C1::Symmetric{T,Matrix{T}},
    C2::Symmetric{T,Matrix{T}},
    f::Forstner,
) where {T<:Real}
    A = pinv(sqrt(C1), f.α)
    L = Symmetric(A * C2 * A') + f.α * I(size(C1, 1))
    vals = eigvals(L)
    sum(log.(abs.(vals)) .^ 2)
end

"""
    Euclidean <: Distance 
        Cinv :: Covariance Matrix 

    Computes the squared euclidean distance with weight matrix Cinv, the inverse of some covariance matrix.
"""
struct Euclidean{T} <: Distance where {T}
    Cinv::Union{Symmetric{T,Matrix{T}},Diagonal{T,Vector{T}}}
    Csqrt::Union{Symmetric{T,Matrix{T}},Diagonal{T,Vector{T}}}
end
function Euclidean(dim::Int)
    Euclidean(1.0 * I(dim), 1.0 * I(dim))
end
function Euclidean(
    Cinv::Union{Symmetric{T,Matrix{T}},Diagonal{T,Vector{T}}},
) where {T<:Real}
    Csqrt = sqrt(Cinv)
    Euclidean(Cinv, Csqrt)
end

"""
    compute_distance(A, B, d)

Compute the distance between features A and B using distance metric d. 
"""
function compute_distance(B1::Vector{T}, B2::Vector{T}, e::Euclidean) where {T<:Real}
    (B1 - B2)' * e.Cinv * (B1 - B2)
end

function compute_distance(
    C1::Symmetric{T,Matrix{T}},
    C2::Symmetric{T,Matrix{T}},
    e::Euclidean,
) where {T<:Real}
    tr(e.Csqrt * (C1 - C2)' * e.Cinv * (C1 - C2) * e.Csqrt)
end

"""
    compute_gradx_distance(A, B, d)

Compute gradient of the distance between features A and B using distance metric d, with respect to the first argument (A). 
"""
function compute_gradx_distance(
    A::T,
    B::T,
    e::Euclidean
    ) where {T<:Vector{<:Real}}

    return 2 * e.Cinv * (A - B)
end

"""
    compute_grady_distance(A, B, d)

Compute gradient of the distance between features A and B using distance metric d, with respect to the second argument (B). 
"""
function compute_grady_distance(
    A::T,
    B::T,
    e::Euclidean
    ) where {T<:Vector{<:Real}}

    return -2 * e.Cinv * (A - B)
end

"""
    compute_gradxy_distance(A, B, d)

Compute second-order cross derivative of the distance between features A and B using distance metric d. 
"""
function compute_gradxy_distance(
    A::T,
    B::T,
    e::Euclidean
    ) where {T<:Vector{<:Real}}

    return -2 * e.Cinv
end