using Statistics
using OrderedCollections

function Base.split(ds, n, m)
    ii = randperm(length(ds))
    return ds[first(ii, n)], ds[last(ii, m)]
end

###

#function get_energy_vals(ds)
#    return get_values.(get_energy.(ds))
#end

#function get_energy_pred_vals(lp, ds)
#    return [B ⋅ lp.β for B in compute_feature.(get_local_descriptors.(ds), [GlobalSum()])]
#end

#function get_forces_vals(ds)
#    return vcat(vcat(get_values.(get_forces.(ds))...)...)
#end

#function get_forces_pred_vals(ds)
#    force_descriptors = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds]
#    return vcat([dB' * lp.β for dB in [reduce(hcat, fi) for fi in force_descriptors]]...)
#end


function get_true_values(ds)
    e = get_values.(get_energy.(ds))
    f = vcat(vcat(get_values.(get_forces.(ds))...)...)
    return e, f
end

function get_pred_values(lp, ds)
    e_pred = [B ⋅ lp.β for B in compute_feature.(get_local_descriptors.(ds), [GlobalSum()])]
    force_descriptors = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds]
    f_pred = vcat([dB' * lp.β for dB in [reduce(hcat, fi) for fi in force_descriptors]]...)
    return e_pred, f_pred
end


###

function get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                      e_test_pred, e_test, f_test_pred, f_test,
                      B_time, dB_time, time_fitting)
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
                            "time_fitting [s]" => time_fitting)
    return metrics
end
