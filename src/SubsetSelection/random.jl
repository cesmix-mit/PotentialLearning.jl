"""
    struct Random
        num_configs :: Int 
        batch_size  :: Int 
    end

A convenience function that allows the user to randomly select indices uniformly over [1, num_configs]. 
"""
struct RandomSelector <: SubsetSelector
    num_configs :: Int 
    batch_size :: Int
end
function RandomSelector(num_configs::Int; batch_size = num_configs)
    RandomSelector(num_configs, batch_size)
end
"""
    get_random_subset(r::Random, batch_size :: Int) <: Vector{Int64}

Access a random subset of the data as sampled from the provided k-DPP. Returns the indices of the random subset and the subset itself.
"""
function get_random_subset(r::RandomSelector, batch_size :: Int = r.batch_size)
    randperm(r.num_configs)[1:batch_size]
end