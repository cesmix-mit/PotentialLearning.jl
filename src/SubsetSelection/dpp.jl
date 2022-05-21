struct kDPP{F1, F2} <:  SubsetSelector
    K   :: EllEnsemble
end
"""
    kDPP(lp::LinearProblem, diversity::Function, quality::Function) <: SubsetSelector

A convenience function that allows the user access to a k-Determinantal Point Process through DPP.jl. All that is required to construct a kDPP is a similarity kernel, for which the user must provide a LinearProblem and two functions to compute descriptor (1) diversity and (2) quality. 
"""
function kDPP(LP::LinearProblem, diversity::Function, quality::Function)
    descriptors = LP.descriptors
    n = length(LP.descriptors)
    K = zeros(n, n)
    for i = 1:n
        K[i, i] = quality(descriptors[i])
        for j = (i+1):n
            K[i, j] = diversity(descriptors[i], descriptors[j]) * quality(descriptors[i]) * quality(descriptors[j])
        end
    end

    kDPP(EllEnsemble(Symmetric(K)))
end
"""
    get_random_subset(configs::Vector{AtomsBase.System}, dpp::kDPP)

Access a random subset of the data as sampled from the provided k-DPP. Returns the indices of the random subset and the subset itself.
"""
function get_random_subset(dpp::kDPP, batch_size :: Int)
    indices = DPP.sample(dpp.K, batch_size)
    return indices
end
"""
    get_dpp_mode(configs::Vector{Atoms})

Access an approximate mode of the k-DPP as calculated by a greedy subset algorithm. See DPP.jl for details.
"""
function get_dpp_mode(dpp::kDPP, batch_size::Int)
    indices = greedy_subset(dpp.K, batch_size)
    return (indices, configs[indices])
end



