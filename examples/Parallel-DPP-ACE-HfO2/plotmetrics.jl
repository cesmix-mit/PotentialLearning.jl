using DataFrames, CSV, Statistics, Plots, StatsBase

function plotmetrics(res_path, metrics_filename)
    metrics = CSV.read("$res_path/$metrics_filename", DataFrame)

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
            metric_q3 = []; metric_q2 = []; metric_q1 = []
            for batch_size in batch_sizes
                ms = metrics[ metrics.method .== method .&&
                              metrics.batch_size .== batch_size , metric]

                # Calculation of mean and standard error
                m = mean(ms)
                se = stdm(ms, m) / sqrt(length(ms))
                push!(metric_means, m)
                push!(metric_se, se)

                # Calculation of quantiles
                qs = quantile(ms, [0.25, 0.5, 0.75])
                push!(metric_q3, qs[3])
                push!(metric_q2, qs[2])
                push!(metric_q1, qs[1])
            end
            plot!(batch_sizes,
                metric_q2,
                #metric_means,
                #ribbon = metric_se,
                ribbon = (metric_q2 .- metric_q1, metric_q3 .- metric_q2),
                #yerror = (metric_q2 .- metric_q1, metric_q3 .- metric_q2),
                color = colors[j],
                fillalpha=.05,
                label=method)
            plot!(batch_sizes,
                #metric_means,
                metric_q2,
                seriestype = :scatter,
                thickness_scaling = 1.35,
                markersize = 3,
                markerstrokewidth = 0,
                markerstrokecolor = :black, 
                markercolor = colors[j],
                label="")
            max = metric == :time ? 1 : 0.8 #maximum(metric_q2) #1
            min = metric == :time ? -0.1 : minimum(metric_q2) * 0.50
            plot!(dpi = 300,
                label = "",
                xscale=:log2,
                #yscale=:log10,
                xticks = (batch_sizes, xticks_label),
                ylim=(min, max),
                xlabel = "Training Dataset Size (Sample Size)",
                ylabel = metric_labels[i])
        end
        plot!(legend=:topright)
        savefig("$res_path/$metric.png")
    end
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


