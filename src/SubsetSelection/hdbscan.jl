using HDBSCAN
struct hDBSCAN <: SubsetSelector
    h ::  HDBSCAN.HdbscanResult
end
"""
    hDBSCAN(lp :: LinearProblem, min_cluster_size::Int) <: SubsetSelector

A convenience function that allows the user access to an HDBSCAN object to compute a clustering of the data. Uses HDBSCAN.jl, which is a wrapper for the python function. Requires that users pass a LinearProblem and minimum cluster size.
"""
function hDBSCAN(lp::LinearProblem, min_cluster_size::Int)
    descriptors = lp.descriptors
    X = reduce(hcat, descriptors)
    h = hdbscan(X'; min_cluster_size = min_cluster_size)
    hDBSCAN(h)
end
function get_random_subset(hdbscan::hDBSCAN, batch_size :: Int)
    indices = Int[]
    assignments = hdbscan.h.assignments
    for i in unique(assignments)
        mini_inds = findall(assignments .== i)
        append!(indices, shuffle(mini_inds)[1:batch_size])
    end
    shuffle(indices)[1:batch_size]
end