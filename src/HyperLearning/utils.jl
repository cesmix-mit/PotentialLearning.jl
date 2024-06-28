
# Estimate force calculation time
function estimate_time(confs, iap; batch_size = 50)
    if length(confs) < batch_size
        batch_size = length(confs)
    end
    random_selector = RandomSelector(length(confs), batch_size)
    inds = PotentialLearning.get_random_subset(random_selector)
    time = @elapsed begin
        f_descr = compute_force_descriptors(confs[inds],
                                            iap.basis,
                                            pbar = false)
        ds = DataSet(confs[inds] .+ f_descr)
        f_pred = get_all_forces(ds, iap)
    end
    n_atoms = sum(length(get_system(c)) for c in confs[inds])
    return time / n_atoms
end

# Get results from the hyperoptimizer
function get_results(ho)
    column_names = string.(vcat(keys(ho.results[1][2])..., ho.params...))
    rows = [[values(r[2])..., p...] for (r, p) in zip(ho.results, ho.history)]
    results = DataFrame([Any[] for _ in 1:length(column_names)], column_names)
    [push!(results, r) for r in rows]
    return sort!(results)
end

