using DataFrames, CSV, Statistics, Plots

metrics = CSV.read("$res_path/metrics.csv", DataFrame)

methods = reverse(unique(metrics.method))
batch_sizes = unique(metrics.batch_size)
batch_size_prop = unique(metrics.batch_size_prop)
xticks_label = ("$b\n$(p*100)%" for (b, p) in zip(batch_sizes, batch_size_prop))
colors = palette(:tab10)
metrics_cols = [:e_train_mae, :f_train_mae, :e_test_mae, :f_test_mae, :time]
metric_labels = ["E MAE | eV/atom",
                 "F MAE | eV/Å",
                 "E MAE | eV/atom",
                 "F MAE | eV/Å",
                 "Time | s"]
for (i, metric) in enumerate(metrics_cols)
    plot()
    for (j, method) in enumerate(methods)
        metric_means = []; metric_se = []
        for batch_size in batch_sizes
            ms = metrics[ metrics.method .== method .&&
                          metrics.batch_size .== batch_size , metric]
            m = mean(ms)
            se = stdm(ms, m) / sqrt(length(ms)) # standard error
            push!(metric_means, m)
            push!(metric_se, se)
        end
        plot!(batch_sizes,
              metric_means,
              ribbon = metric_se,
              color = colors[j],
              fillalpha=.1,
              label=method)
        plot!(batch_sizes,
              metric_means,
              seriestype = :scatter,
              thickness_scaling = 1.35,
              markersize = 3,
              markerstrokewidth = 0,
              markerstrokecolor = :black, 
              markercolor = colors[j],
              label="")
        max = metric == :time ? 1 : 1
        min = metric == :time ? -1 : minimum(metric_means) * 0.99
        plot!(dpi = 300,
              label = "",
              xscale=:log2, 
              xticks = (batch_sizes, xticks_label),
              ylim=(min, max),
              xlabel = "Training Dataset Sample Size",
              ylabel = metric_labels[i])
    end
    savefig("$res_path/$metric.png")
end


# xformatter = :scientific,
#              markershape = :circle,
#              markercolor = :gray
# yerror=metric_std,
#ribbon=metric_std,
#yerror=metric_std,
# markerstrokewidth=0, markersize=5, 
#yaxis=:log,
#xaxis=:log2, yaxis=:log,

#for metric in [:e_train_mae, :f_train_mae, :e_test_mae, :f_test_mae, :time]
#    scatter()
#    for method in reverse(unique(metrics[:, :method])[1:end])
#        batch_size_vals = metrics[metrics.method .== method, :][:, :batch_size]
#        metric_vals = metrics[metrics.method .== method, :][:, metric]
#        scatter!(batch_size_vals, metric_vals, label = method,
#                 alpha = 0.5, dpi=300, markerstrokewidth=0, markersize=5, xaxis=:log2, yaxis=:log,
#                 xlabel = "Sample size",
#                 ylabel = "$metric")
#    end
#    savefig("$res_path/$metric-srs.png")
#end

#scatter()
#for method in reverse(unique(metrics[:, :method])[2:end])
#    batch_size_vals = metrics[metrics.method .== method, :][:, :batch_size]
#    speedup_vals = metrics[metrics.method .== "DPP", :][:, :time] ./
#                  metrics[metrics.method .== method, :][:, :time]
#    scatter!(batch_size_vals, speedup_vals, label = "DPP time / $method time",
#             alpha = 0.5, dpi=300, markerstrokewidth=0, markersize=5, xaxis=:log2,
#             xlabel = "Sample size",
#             ylabel = "Speedup")
#end
#savefig("$res_path/speedup-srs.png")



#using DataFrames, CSV, Plots

#metrics = CSV.read("metrics.csv", DataFrame)
#res_path = "dyomet/"

#for metric in [:e_train_mae, :f_train_mae, :e_test_mae, :f_test_mae, :time]
#    scatter()
#    for method in reverse(unique(metrics[:, :method])[1:end])
#        batch_size_vals = metrics[metrics.method .== method, :][:, :batch_size]
#        metric_vals = metrics[metrics.method .== method, :][:, metric]
#        scatter!(batch_size_vals, metric_vals, label = method,
#                 alpha = 0.5, dpi=300, markerstrokewidth=0, markersize=5, xaxis=:log2, yaxis=:log,
#                 xlabel = "Sample size",
#                 ylabel = "$metric")
#    end
#    savefig("$res_path/$metric-srs.png")
#end

#scatter()
#for method in reverse(unique(metrics[:, :method])[2:end])
#    batch_size_vals = metrics[metrics.method .== method, :][:, :batch_size]
#    speedup_vals = metrics[metrics.method .== "DPP", :][:, :time] ./
#                  metrics[metrics.method .== method, :][:, :time]
#    scatter!(batch_size_vals, speedup_vals, label = "DPP time / $method time",
#             alpha = 0.5, dpi=300, markerstrokewidth=0, markersize=5, xaxis=:log2,
#             xlabel = "Sample size",
#             ylabel = "Speedup")
#end
#savefig("$res_path/speedup-srs.png")


