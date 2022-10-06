include("features.jl")
include("distances.jl")

export Distance, Forstner, compute_distance, Euclidean
export Feature, GlobalMean, CorrelationMatrix, compute_feature, compute_features
export Kernel, DotProduct, get_parameters, RBF, compute_kernel, KernelMatrix
###############
"""
    abstract type Kernel end 

    A struct of abstract type Kernel is function that takes in two features and produces a semi-definite scalar representing the similarity between the two features.
"""
abstract type  Kernel end

"""
    struct DotProduct <: Kernel 
        α :: Power of DotProduct kernel 
    end

    Computes the dot product kernel between two features, i.e.,

    cos(θ) = ( A ⋅ B / (||A||^2||B||^2) )^α
"""
struct DotProduct <: Kernel 
    α :: Int
end 
function DotProduct(; α = 2)
    DotProduct(α)
end

get_parameters(k::DotProduct) = (k.α,)

function compute_kernel(A::T, B::T, d::DotProduct) where T <: Union{Vector{<:Real}, Symmetric{<:Real, <:Matrix{<:Real}}}
    ( dot(A, B) / sqrt(dot(A, A) * dot(B, B)) )^d.α
end

##
""" 
    struct RBF <: Kernel 
        d :: Distance function 
        α :: Reguarlization parameter 
        ℓ :: Length-scale parameter
        β :: Scale parameter
    end

    Computes the squared exponential kernel, i.e.,

     k(A, B) = β \exp( -\frac{1}{2} d(A,B)/ℓ^2 ) + α δ(A, B) 
"""
struct RBF <: Kernel 
    d :: Distance 
    α :: Real
    ℓ :: Real 
    β :: Real
end
function RBF(d; α = 1e-8, ℓ = 1.0, β = 1.0)
   RBF(d, α, ℓ, β) 
end

get_parameters(k::RBF) = (k.α, k.ℓ, k.β)

"""
    function compute_kernel(A, B, k) end 

Compute similarity kernel between features A and B using kernel k. 
"""
function compute_kernel(A::T, B::T, r::RBF) where T <: Union{Vector{<:Real}, Symmetric{<:Real, <:Matrix{<:Real}}}
    d2 = compute_distance(A, B, r.d)
    r.β*exp(-0.5*d2/r.ℓ)
end

""" 
    function KernelMatrix(F, k::Kernel)

Compute symmetric kernel matrix K where K_{ij} = k(F_i, F_j). 
"""
function KernelMatrix(F::Union{Vector{Vector{T}}, Vector{Symmetric{T, Matrix{T}}}}, k::Kernel) where T
    n = length(F)
    K = zeros(n, n)
    for i = 1:n
        for j = 1:n 
            K[i, j] = compute_kernel(F[i], F[j], k)
        end
    end
    Symmetric(K)
end
""" 
    function KernelMatrix(F1, F2, k::Kernel)

Compute non-symmetric kernel matrix K where K_{ij} = k(F1_i, F2_j). 
"""
function KernelMatrix(F1::Union{Vector{Vector{T}}, Vector{Symmetric{T, Matrix{T}}}}, 
                      F2::Union{Vector{Vector{T}}, Vector{Symmetric{T, Matrix{T}}}}, 
                      k::Kernel) where T
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
    function KernelMatrix(ds::DataSet, F::Feature, k::Kernel)

Compute symmetric kernel matrix K using features of the dataset ds calculated using the Feature method F. 
"""
function KernelMatrix(ds::DataSet, f::Feature, k::Kernel; dt = LocalDescriptors) 
    F = compute_feature.(ds, (f,); dt = dt)
    KernelMatrix(F, k)
end
""" 
    function KernelMatrix(ds1::DataSet, ds2::DataSet, F::Feature, k::Kernel)

    Compute nonsymmetric kernel matrix K using features of the datasets ds1 and ds2 calculated using the Feature method F. 
"""
function KernelMatrix(ds1::DataSet, ds2::DataSet, f::Feature, k::Kernel; dt = LocalDescriptors)
    F1 = compute_feature.(ds1, (f,); dt = dt)
    F2 = compute_feature.(ds2, (f,); dt = dt)
    KernelMatrix(F1, F2, k)
end