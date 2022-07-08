# Calculate metrics
function calc_metrics(x_pred, x)
    x_mae = sum(abs.(x_pred .- x)) / length(x)
    x_rmse = sqrt(sum((x_pred .- x).^2) / length(x))
    x_rsq = 1 - sum((x_pred .- x).^2) / sum((x .- mean(x)).^2)
    return x_mae, x_rmse, x_rsq
end

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
