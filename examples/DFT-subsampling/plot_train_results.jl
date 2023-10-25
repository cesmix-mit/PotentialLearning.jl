push!(Base.LOAD_PATH, "../../")

using PotentialLearning
using LinearAlgebra, Statistics, StatsBase
using CairoMakie, Colors
using DataFrames
using JLD



# load dadta ------------------------------------------------------------------------------
elname = "Hf"
param = [5,6]
batch_sets = [10000, 5000, 1000, 500, 100, 50]
exp_dir = "./DPP_training/$elname/ACE_$(param[1])body_$(param[2])deg/"

res_dpp = JLD.load(exp_dir*"DPP_training_results_all.jld")["res"]
res_srs = JLD.load(exp_dir*"SRS_training_results_all.jld")["res"]


df_dpp = compute_cv_metadata(res_dpp)
df_srs = compute_cv_metadata(res_srs)

labels = ["DPP", "SRS"]

# plot metadata
f1 = plot_metadata(df_dpp, df_srs, "E", "rmse", labels)



function plot_metadata(
    df1::DataFrame,
    df2::DataFrame,
    output::String,
    metric::String,
    labels=Vector{String}

)
    batches = df1[:,"batch size"]
    type = "$output $metric"

    with_theme(custom_theme) do
        f = Figure(resolution=(800,600))
        ax = Axis(f[1,1], xlabel="sample size (N)", ylabel=type, yscale=log10, xscale=log10)

        scatterlines!(ax, batches, df1[:,type*" mean"], color=:firebrick, label=labels[1])
        # band!(ax, batches,
        #     df1[:,type*" mean"] .- 2*df1[:,type*" std"],
        #     df1[:,type*" mean"] .+ 2*df1[:,type*" std"],
        #     color=(:firebrick, 0.3)
        # )

        
        scatterlines!(ax, batches, df2[:,type*" mean"], color=:skyblue, label=labels[2])
        # band!(ax, batches,
        #     df2[:,type*" mean"] .- 2*df2[:,type*" std"],
        #     df2[:,type*" mean"] .+ 2*df2[:,type*" std"],
        #     color=(:skyblue, 0.3)
        # )

        axislegend(ax)
        return f
    end
end




custom_theme = Theme(
    font = "/Users/joannazou/Documents/MIT/Presentations/UNCECOMP 2023/InterFont/static/Inter-Regular.ttf",
    fontsize = 20,
    Axis = (
        xgridcolor = :white,
        ygridcolor = :white,
    )
)

