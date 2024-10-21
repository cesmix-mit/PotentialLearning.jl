#################################################################################
# Find best sampling methods for a given atomistic dataset
#################################################################################

# Load packages #################################################################
using AtomsBase, InteratomicPotentials, PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra, Random, DisplayAs
using Statistics, Distances
using Determinantal, Clustering
using MultivariateStats, LowRankApprox
using DataFrames, Plots
using Hyperopt
using MPI
using JLD
using Colors
using Base.Threads

MPI.Init()
comm = MPI.COMM_WORLD
size = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

# Define paths and create experiment folder ###################################
base_path = haskey(ENV, "BASE_PATH") ? ENV["BASE_PATH"] : "../../"
ds_path   = "$base_path/examples/data/Hf/"
res_path  = "$base_path/examples/Parallel-DPP-ACE-HfO2/results-Hf/";

run(`mkdir -p $res_path`);

# Load auxiliary functions  ####################################################
include("$base_path/examples/utils/utils.jl")
include("$base_path/examples/Parallel-DPP-ACE-HfO2/samplers.jl")
include("$base_path/examples/Parallel-DPP-ACE-HfO2/aux_sample_functions.jl")
include("$base_path/examples/Parallel-DPP-ACE-HfO2/plotmetrics.jl")

# Load training and test configuration datasets ################################

# Dataset 1 (28k)
paths = ["$ds_path/Hf2_gas_form_sorted.extxyz",
         "$ds_path/Hf2_mp103_EOS_1D_form_sorted.extxyz", # 200
         "$ds_path/Hf2_mp103_EOS_3D_form_sorted.extxyz", # 9377
         "$ds_path/Hf2_mp103_EOS_6D_form_sorted.extxyz", # 17.2k
         "$ds_path/Hf128_MC_rattled_mp100_form_sorted.extxyz", # 306
         "$ds_path/Hf128_MC_rattled_mp103_form_sorted.extxyz", # 50
         "$ds_path/Hf128_MC_rattled_random_form_sorted.extxyz", # 498
         "$ds_path/Hf_mp100_EOS_1D_form_sorted.extxyz", # 201
         "$ds_path/Hf_mp100_primitive_EOS_1D_form_sorted.extxyz"
        ]

# Dataset 2
#paths = [
#         "$ds_path/HfO2_figshare_form_sorted.extxyz",
#         "$ds_path/HfO2_mp550893_EOS_1D_form_sorted.extxyz",
#         "$ds_path/HfO_gas_form_sorted.extxyz",
#         "$ds_path/HfO2_figshare_form_sorted.extxyz",
#         "$ds_path/HfO2_mp352_EOS_1D_form_sorted.extxyz",
#         "$ds_path/HfO2_mp550893_EOS_6D_form_sorted.extxyz",
#         "$ds_path/Hf2_gas_form_sorted.extxyz",
#         "$ds_path/Hf2_mp103_EOS_1D_form_sorted.extxyz",
#         "$ds_path/Hf2_mp103_EOS_3D_form_sorted.extxyz",
#         "$ds_path/Hf2_mp103_EOS_6D_form_sorted.extxyz",
#         "$ds_path/Hf_mp100_EOS_1D_form_sorted.extxyz", 
#         "$ds_path/Hf128_MC_rattled_mp100_form_sorted.extxyz",
#         "$ds_path/Hf128_MC_rattled_mp103_form_sorted.extxyz",
#         "$ds_path/Hf128_MC_rattled_random_form_sorted.extxyz",
#         "$ds_path/Hf_mp100_primitive_EOS_1D_form_sorted.extxyz",
#]

confs = []
for ds_path in paths
    push!(confs, load_data(ds_path, uparse("eV"), uparse("Å"))...)
end
confs = DataSet(confs)
n = length(confs)
GC.gc()

#ds_path = string("$ds_path/a-HfO2-300K-NVT-6000.extxyz")
#confs = load_data(ds_path, uparse("eV"), uparse("Å"))
#n = length(confs)

species = unique(vcat([atomic_symbol.(get_system(c).particles)
          for c in confs]...))

# Compute ACE descriptors to update dataset
basis = ACE(species           = species,
            body_order        = 8,
            polynomial_degree = 8,
            rcutoff           = 5.5,
            wL                = 1.0,
            csp               = 1.0,
            r0                = 1.0)
@save_var res_path basis

# Update training dataset by adding energy and force descriptors
println("Computing energy descriptors of dataset...")
B_time = @elapsed e_descr = compute_local_descriptors(confs, basis)
println("Computing force descriptors of dataset...")
dB_time = @elapsed f_descr = compute_force_descriptors(confs, basis)
GC.gc()
ds = DataSet(confs .+ e_descr .+ f_descr)

# Subsampling experiments #####################################################

# Define number of experiments
n_experiments = 10

# Define samplers
#samplers = [simple_random_sample, dbscan_sample, kmeans_sample, 
#            cur_sample, dpp_sample, lrdpp_sample]
samplers = [simple_random_sample, kmeans_sample, cur_sample, lrdpp_sample]

# Define batch sample sizes (proportions)
batch_size_props = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64] # TODO: add 0.99 in final experiment

# Define precisions
# precs = [Float32, Float64]

# Create metric dataframe
metric_names = [:exp_number,  :method, :batch_size_prop, :batch_size, :time,
                :e_train_mae, :e_train_rmse, :e_train_rsq,
                :f_train_mae, :f_train_rmse, :f_train_rsq, :f_train_mean_cos,
                :e_test_mae,  :e_test_rmse,  :e_test_rsq, 
                :f_test_mae,  :f_test_rmse,  :f_test_rsq,  :f_test_mean_cos]
metrics = DataFrame([Any[] for _ in 1:length(metric_names)], metric_names)

# Run experiments
local_exp = ceil(Int, n_experiments / size)
for nc in 1:local_exp
    #check it there is left over
    j = rank + size * (nc-1) + 1
    if j > n_experiments
        break
    end
    global metrics
    
    # Define randomized training and test dataset
    n_train = floor(Int, 0.8 * n)
    n_test = n - n_train
    rnd_inds = randperm(n)
    rnd_inds_train = rnd_inds[1:n_train]
    rnd_inds_test = rnd_inds[n_train+1:n_train+n_test] # rnd_inds[n_train+1:end]
    ds_train_rnd = @views ds[rnd_inds_train]
    ds_test_rnd  = @views ds[rnd_inds_test]
    ged = sum.(get_values.(get_local_descriptors.(ds_train_rnd)))
    ged_mat = stack(ged)'

    # Sampling experiments
    for batch_size_prop in batch_size_props
        for sampler in samplers
            sample_experiment!(res_path, j, sampler, batch_size_prop, n_train, 
                               ged_mat, ds_train_rnd, ds_test_rnd, basis, metrics)
            GC.gc()
        end
    end
end

# Postprocess ##################################################################
plotmetrics("$res_path/metrics.csv")
