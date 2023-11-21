push!(Base.LOAD_PATH, "../../")

using PotentialLearning
using LinearAlgebra, Statistics, StatsBase
using CairoMakie, Colors
using DataFrames
using JLD

include("subsampling_utils.jl")


# load dadta ------------------------------------------------------------------------------
elname = "Hf"
param = [5,6]
# batch_sets = [250, 500, 1000, 2000, 4000, 8000, 16000]
exp_dir = "./DPP_training/$elname/ACE_$(param[1])body_$(param[2])deg/"

res_dpp = JLD.load(exp_dir*"DPP_training_results_MC_all.jld")["res"]
res_srs = JLD.load(exp_dir*"SRS_training_results_MC_all.jld")["res"]
batch_sets = collect(keys(res_dpp))

df_dpp = compute_cv_metadata(res_dpp)
df_srs = compute_cv_metadata(res_srs)
delete!(df_dpp, 1)
delete!(df_srs, 1)

labels = ["DPP", "SRS"]

# plot metadata
f1 = plot_metadata(df_dpp, df_srs, "E", "mae", labels) # minmax="min")
f3 = plot_metadata(df_dpp, df_srs, "E", "rsq", labels) # ; minmax="max")
f4 = plot_metadata(df_dpp, df_srs, "F", "mae", labels) # ; minmax="min")
f5 = plot_metadata(df_dpp, df_srs, "F", "rmse", labels) # minmax="min")
f6 = plot_metadata(df_dpp, df_srs, "F", "rsq", labels) # minmax="max")


function plot_metadata(
    df1::DataFrame,
    df2::DataFrame,
    output::String,
    metric::String,
    labels=Vector{String};
    minmax::Union{Bool,String}=false
)
    batches = df1[:,"batch size"]

    type = "$output $metric"

    f = Figure(resolution=(500,500))
    ax = Axis(f[1,1],
        xlabel="sample size (N)",
        ylabel=type,
        xscale=log10,
        yscale=log10,
        xticks=(batches, [string(bs) for bs in sort(batches)]),
        # xgridvisible=false,
        )

    if minmax == false
        scatter!(ax, batches, abs.(df1[:,type*" med"]), color=:firebrick, marker=:rect, markersize=15, label=labels[1])
        rangebars!(ax, batches,
            abs.(df1[:,type*" lqt"]),
            abs.(df1[:,type*" uqt"]),
            color=:firebrick,
            linewidth=3,
        )
        
        scatter!(ax, batches, abs.(df2[:,type*" med"]), color=:skyblue, markersize=15, label=labels[2])
        rangebars!(ax, batches,
            abs.(df2[:,type*" lqt"]),
            abs.(df2[:,type*" uqt"]),
            color=:skyblue,
            linewidth=3,
        )
    elseif typeof(minmax) == String
        scatter!(ax, batches, df1[:,type*" $minmax"], color=:firebrick, label=labels[1])
        scatter!(ax, batches, df2[:,type*" $minmax"], color=:skyblue, label=labels[2])
    end
    axislegend(ax, position=:rb)
    return f
end






custom_theme = Theme(
    font = "/Users/joannazou/Documents/MIT/Presentations/UNCECOMP 2023/InterFont/static/Inter-Regular.ttf",
    fontsize = 20,
    Axis = (
        xgridcolor = :white,
        ygridcolor = :white,
    )
)

