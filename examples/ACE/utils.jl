using Statistics
using OrderedCollections

# Split datasets

function Base.split(ds, n, m)
    ii = randperm(length(ds))
    return ds[first(ii, n)], ds[last(ii, m)]
end

# Auxiliary functions to compute all energies and forces as vectors (Zygote-friendly functions)

function get_all_energies(ds::DataSet)
    return [get_values(get_energy(ds[c])) for c in 1:length(ds)]
end

function get_all_forces(ds::DataSet)
    return reduce(vcat,reduce(vcat,[get_values(get_forces(ds[c]))
                                    for c in 1:length(ds)]))
end

function get_all_energies(ds::DataSet, lp::PotentialLearning.LinearProblem)
    Bs = sum.(get_values.(get_local_descriptors.(ds)))
    return dot.(Bs, [lp.β])
end

function get_all_forces(ds::DataSet, lp::PotentialLearning.LinearProblem)
    force_descriptors = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds]
    return vcat([dB' * lp.β for dB in [reduce(hcat, fi) for fi in force_descriptors]]...)
end

function get_all_energies(ds::DataSet, nnbp::NNBasisPotential)
    return [potential_energy(ds[c], nnbp) for c in 1:length(ds)]
end

function get_all_forces(ds::DataSet, nnbp::NNBasisPotential)
    return reduce(vcat,reduce(vcat,[force(ds[c], nnbp)
                                    for c in 1:length(ds)]))
end

# Metrics

function get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                      e_test_pred, e_test, f_test_pred, f_test,
                      B_time, dB_time, learn_time)
    e_train_mae, e_train_rmse, e_train_rsq = calc_metrics(e_train_pred, e_train)
    f_train_mae, f_train_rmse, f_train_rsq = calc_metrics(f_train_pred, f_train)
    e_test_mae, e_test_rmse, e_test_rsq = calc_metrics(e_test_pred, e_test)
    f_test_mae, f_test_rmse, f_test_rsq = calc_metrics(f_test_pred, f_test)
    
    f_test_pred_v = collect(eachcol(reshape(f_test_pred, 3, :)))
    f_test_v = collect(eachcol(reshape(f_test, 3, :)))
    f_test_cos = dot.(f_test_v, f_test_pred_v) ./ (norm.(f_test_v) .* norm.(f_test_pred_v))
    f_test_mean_cos = mean(f_test_cos)

    metrics = OrderedDict(  "e_train_mae"      => e_train_mae,
                            "e_train_rmse"     => e_train_rmse,
                            "e_train_rsq"      => e_train_rsq,
                            "f_train_mae"      => f_train_mae,
                            "f_train_rmse"     => f_train_rmse,
                            "f_train_rsq"      => f_train_rsq,
                            "e_test_mae"       => e_test_mae,
                            "e_test_rmse"      => e_test_rmse,
                            "e_test_rsq"       => e_test_rsq,
                            "f_test_mae"       => f_test_mae,
                            "f_test_rmse"      => f_test_rmse,
                            "f_test_rsq"       => f_test_rsq,
                            "f_test_mean_cos"  => f_test_mean_cos,
                            "B_time [s]"       => B_time,
                            "dB_time [s]"      => dB_time,
                            "learn_time [s]" => learn_time)
    return metrics
end
