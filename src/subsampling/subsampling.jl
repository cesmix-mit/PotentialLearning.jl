
export random_subsample

"""
    random_subsample(systems, energies, forces, stress, n_sys)
    
`systems`: vector with atomic configurations
`energies`: vector with the energies of each atomic configuration
`forces`: vector with the forces of each atomic configuration
`stress`: vector with the stress of each atomic configuration
`max_sys`: maximum random subset size.

Returns a random subset of the input vectors.

"""
function random_subsample(systems, energies, forces, stress; max_sys = length(systems))
    n = length(systems)
    max_sys = min(max_sys, n)
    rand_list = randperm(n)[1:max_sys]
    return systems[rand_list], energies[rand_list], forces[rand_list], stress[rand_list]
end


# TODO: DPP. See branch https://github.com/cesmix-mit/PotentialLearning.jl/tree/active_learning_refactor

# TODO: See https://docs.google.com/document/d/1SWAanEWQkpsbr2lqetMO3uvdX_QK-Z7dwrgPaM1Dl0o/edit?usp=sharing
