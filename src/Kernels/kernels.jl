include("features.jl")
include("distances.jl")

###############
"""
    Kernel

    A struct of abstract type Kernel is function that takes in two features and produces a semi-definite scalar representing the similarity between the two features.
"""
abstract type Kernel end

"""
    DotProduct <: Kernel 
        α :: Power of DotProduct kernel 


    Computes the dot product kernel between two features, i.e.,

    cos(θ) = ( A ⋅ B / (||A||^2||B||^2) )^α
"""
struct DotProduct <: Kernel
    α::Int
end
function DotProduct(; α = 2)
    DotProduct(α)
end

get_parameters(k::DotProduct) = (k.α,)

function compute_kernel(
    A::T,
    B::T,
    d::DotProduct,
) where {T<:Union{Vector{<:Real},Symmetric{<:Real,<:Matrix{<:Real}}}}
    (dot(A, B) / sqrt(dot(A, A) * dot(B, B)))^d.α
end

##
""" 
    RBF <: Kernel 
        d :: Distance function 
        α :: Regularization parameter 
        ℓ :: Length-scale parameter
        β :: Scale parameter
    

    Computes the squared exponential kernel, i.e.,

     k(A, B) = β \exp( -\frac{1}{2} d(A,B)/ℓ^2 ) + α δ(A, B) 
"""
mutable struct RBF <: Kernel
    d::Distance
    α::Real
    ℓ::Real
    β::Real
end
function RBF(d; α = 1e-8, ℓ = 1.0, β = 1.0)
    RBF(d, α, ℓ, β)
end

get_parameters(k::RBF) = (k.α, k.ℓ, k.β)

"""
    compute_kernel(A, B, k)

Compute similarity kernel between features A and B using kernel k. 
"""
function compute_kernel(
    A::T,
    B::T,
    r::RBF,
) where {T<:Union{Vector{<:Real},Symmetric{<:Real,<:Matrix{<:Real}}}}
    d2 = compute_distance(A, B, r.d)
    r.β * exp(-0.5 * d2 / r.ℓ^2)
end

"""
    compute_gradx_kernel(A, B, k)

Compute gradient of the kernel between features A and B using kernel k, with respect to the first argument (A). 
"""
function compute_gradx_kernel(
    A::T,
    B::T,
    r::RBF,
    ) where {T<:Vector{<:Real}}

    k = compute_kernel(A, B, r)
    ∇d = compute_gradx_distance(A, B, r.d)
    return -0.5 * k * ∇d / r.ℓ^2
end

"""
    compute_grady_kernel(A, B, k)

Compute gradient of the kernel between features A and B using kernel k, with respect to the second argument (B). 
"""
function compute_grady_kernel(
    A::T,
    B::T,
    r::RBF,
    ) where {T<:Vector{<:Real}}

    k = compute_kernel(A, B, r)
    ∇d = compute_grady_distance(A, B, r.d)
    return -0.5 * k * ∇d / r.ℓ^2
end

"""
    compute_gradxy_kernel(A, B, k)

Compute the second-order cross derivative of the kernel between features A and B using kernel k. 
"""
function compute_gradxy_kernel(
    A::T,
    B::T,
    r::RBF,
    ) where {T<:Vector{<:Real}}

    k = compute_kernel(A, B, r)
    ∇xd = compute_gradx_distance(A, B, r.d)
    ∇yd = compute_grady_distance(A, B, r.d)
    ∇xyd = compute_gradxy_distance(A, B, r.d)

    return k .* ( -0.5 * ∇xyd / r.ℓ^2 .+ 0.25 * ∇xd'*∇yd / r.ℓ^4 )
    
end

""" 
    KernelMatrix(F, k::Kernel)

Compute symmetric kernel matrix K where K_{ij} = k(F_i, F_j). 
"""
function KernelMatrix(
    F::Union{Vector{Vector{T}},Vector{Symmetric{T,Matrix{T}}}},
    k::Kernel,
) where {T}
    n = length(F)
    K = zeros(n, n)
    for i = 1:n
        for j = i:n
            K[i, j] = compute_kernel(F[i], F[j], k)
        end
    end
    Symmetric(K)
end
""" 
    KernelMatrix(F1, F2, k::Kernel)

Compute non-symmetric kernel matrix K where K_{ij} = k(F1_i, F2_j). 
"""
function KernelMatrix(
    F1::Union{Vector{Vector{T}},Vector{Symmetric{T,Matrix{T}}}},
    F2::Union{Vector{Vector{T}},Vector{Symmetric{T,Matrix{T}}}},
    k::Kernel,
) where {T}
    m = length(F1)
    n = length(F2)
    K = zeros(m, n)
    for i = 1:m
        for j = 1:n
            K[i, j] = compute_kernel(F1[i], F2[j], k)
        end
    end
    K

end
""" 
    KernelMatrix(ds::DataSet, F::Feature, k::Kernel)

Compute symmetric kernel matrix K using features of the dataset ds calculated using the Feature method F. 
"""
function KernelMatrix(ds::DataSet, f::Feature, k::Kernel; dt = LocalDescriptors)
    F = compute_feature.(ds, (f,); dt = dt)
    KernelMatrix(F, k)
end
""" 
    KernelMatrix(ds1::DataSet, ds2::DataSet, F::Feature, k::Kernel)

    Compute nonsymmetric kernel matrix K using features of the datasets ds1 and ds2 calculated using the Feature method F. 
"""
function KernelMatrix(
    ds1::DataSet,
    ds2::DataSet,
    f::Feature,
    k::Kernel;
    dt = LocalDescriptors,
)
    F1 = compute_feature.(ds1, (f,); dt = dt)
    F2 = compute_feature.(ds2, (f,); dt = dt)
    KernelMatrix(F1, F2, k)
end




include("divergences.jl")
export
    Distance,
    Forstner,
    compute_distance,
    compute_gradx_distance,
    compute_grady_distance,
    compute_gradxy_distance,
    Euclidean,
    Feature,
    GlobalMean,
    CorrelationMatrix,
    compute_feature,
    compute_features,
    Kernel,
    DotProduct,
    get_parameters,
    RBF,
    compute_kernel,
    compute_gradx_kernel,
    compute_grady_kernel,
    compute_gradxy_kernel,
    KernelMatrix,
    Divergence,
    KernelSteinDiscrepancy,
    compute_divergence