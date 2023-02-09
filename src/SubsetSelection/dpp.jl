using DPP
"""
    struct kDPP
        K :: EllEnsemble
    end

A convenience function that allows the user access to a k-Determinantal Point Process through DPP.jl. All that is required to construct a kDPP is a similarity kernel, for which the user must provide a LinearProblem and two functions to compute descriptor (1) diversity and (2) quality. 
"""
struct kDPP <: SubsetSelector
    K::EllEnsemble
    batch_size::Int
end
"""
    kDPP(ds::Dataset, f::Feature, k::Kernel) 

A convenience function that allows the user access to a k-Determinantal Point Process through DPP.jl. All that is required to construct a kDPP is a dataset, a method to compute features, and a kernel. Optional arguments include batch size and type of descriptor (default LocalDescriptors).
"""
function kDPP(
    ds::DataSet,
    f::Feature,
    k::Kernel;
    batch_size = length(ds) ÷ 2,
    dt = LocalDescriptors,
)
    K = KernelMatrix(ds, f, k; dt = dt)
    ell = EllEnsemble(K)
    rescale!(ell, batch_size)
    kDPP(ell, batch_size)
end
"""
    kDPP(features::Union{Vector{Vector{T}}, Vector{Symmetric{T, Matrix{T}}}}, k::Kernel) 

A convenience function that allows the user access to a k-Determinantal Point Process through DPP.jl. All that is required to construct a kDPP are features (either a vector of vector features or a vector of symmetric matrix features) and a kernel. Optional argument is batch_size (default length(features)).
"""
function kDPP(
    features::Union{Vector{Vector{T}},Vector{Symmetric{T,Matrix{T}}}},
    k::Kernel;
    batch_size = length(features) ÷ 2,
) where {T}
    K = KernelMatrix(features, k)
    ell = EllEnsemble(K)
    rescale!(ell, batch_size)
    kDPP(ell, batch_size)
end
"""
    get_random_subset(dpp::kDPP, batch_size :: Int) <: Vector{Int64}

Access a random subset of the data as sampled from the provided k-DPP. Returns the indices of the random subset and the subset itself.
"""
function get_random_subset(dpp::kDPP; batch_size::Int = dpp.batch_size)
    indices = DPP.sample(dpp.K, batch_size)
    return indices
end
"""
    get_dpp_mode(dpp::kDPP, batch_size::Int) <: Vector{Int64}

Access an approximate mode of the k-DPP as calculated by a greedy subset algorithm. See DPP.jl for details.
"""
function get_dpp_mode(dpp::kDPP; batch_size::Int = dpp.batch_size)
    indices = greedy_subset(dpp.K, batch_size)
    return indices
end
"""
    get_inclusion_prob(dpp::kDPP) <: Vector{Float64}

Access an approximation to the inclusion probabilities as calculated by DPP.jl (see package for details).
"""
function get_inclusion_prob(dpp::kDPP)
    vec(DPP.inclusion_prob(dpp.K))
end
