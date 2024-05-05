# Run this script:
#   $ cd examples/ACE
#   $ julia --project=../ --threads=4
#   julia> include("fit-ace.jl")

push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using Random
using DataFrame
include("../utils/utils.jl")

# Fit ACE and postprocess results. Used in subsampling experiments #############

function fit(path, ds_train, ds_test, basis)

    # Learn
    lb = LBasisPotential(basis)
    ws, int = [1.0, 1.0], false
    learn!(lb, ds_train, ws, int)

    @save_var path lb.β
    @save_var path lb.β0

    # Post-process output: calculate metrics, create plots, and save results #######

    # Get true and predicted values
    n_atoms_train = length.(get_system.(ds_train))
    n_atoms_test = length.(get_system.(ds_test))

    e_train, e_train_pred = get_all_energies(ds_train) ./ n_atoms_train,
                            get_all_energies(ds_train, lb) ./ n_atoms_train
    f_train, f_train_pred = get_all_forces(ds_train),
                            get_all_forces(ds_train, lb)
    @save_var path e_train
    @save_var path e_train_pred
    @save_var path f_train
    @save_var path f_train_pred

    e_test, e_test_pred = get_all_energies(ds_test) ./ n_atoms_test,
                          get_all_energies(ds_test, lb) ./ n_atoms_test
    f_test, f_test_pred = get_all_forces(ds_test),
                          get_all_forces(ds_test, lb)
    @save_var path e_test
    @save_var path e_test_pred
    @save_var path f_test
    @save_var path f_test_pred

    # Compute metrics
    e_train_metrics = get_metrics(e_train, e_train_pred,
                                  metrics = [mae, rmse, rsq],
                                  label = "e_train")
    f_train_metrics = get_metrics(f_train, f_train_pred,
                                  metrics = [mae, rmse, rsq, mean_cos],
                                  label = "f_train")
    train_metrics = merge(e_train_metrics, f_train_metrics)
    @save_dict path train_metrics

    e_test_metrics = get_metrics(e_test, e_test_pred,
                                 metrics = [mae, rmse, rsq],
                                 label = "e_test")
    f_test_metrics = get_metrics(f_test, f_test_pred,
                                 metrics = [mae, rmse, rsq, mean_cos],
                                 label = "f_test")
    test_metrics = merge(e_test_metrics, f_test_metrics)
    @save_dict path test_metrics

    # Plot and save results

    e_plot = plot_energy(e_train, e_train_pred,
                         e_test, e_test_pred)
    @save_fig path e_plot

    f_plot = plot_forces(f_train, f_train_pred,
                         f_test, f_test_pred)
    @save_fig path f_plot

    e_train_plot = plot_energy(e_train, e_train_pred)
    f_train_plot = plot_forces(f_train, f_train_pred)
    f_train_cos  = plot_cos(f_train, f_train_pred)
    @save_fig path e_train_plot
    @save_fig path f_train_plot
    @save_fig path f_train_cos

    e_test_plot = plot_energy(e_test, e_test_pred)
    f_test_plot = plot_forces(f_test, f_test_pred)
    f_test_cos  = plot_cos(f_test, f_test_pred)
    @save_fig path e_test_plot
    @save_fig path f_test_plot
    @save_fig path f_test_cos
    
    return e_train_metrics, f_train_metrics, 
           e_test_metrics, f_test_metrics
end


# Load training and test configuration datasets ################################

ds_path = "../data/HfO2/"

ds_path = string("../data/HfO2_large/train/HfO2_figshare_form_random_train.extxyz")
conf_train = load_data(ds_path, uparse("eV"), uparse("Å"))
n_train = length(conf_train)

ds_path = string("../data/HfO2_large/test/HfO2_figshare_form_random_test.extxyz")
conf_test = load_data(ds_path, uparse("eV"), uparse("Å"))
n_test = length(conf_test)

n_train, n_test = length(conf_train), length(conf_test)

species = unique(vcat([atomic_symbol.(get_system(c).particles)
          for c in conf_train]...))

path = "full-vs-split-subsampling/"
run(`mkdir -p $path`)

# Compute descriptors ##########################################################

# Compute ACE descriptors
basis = ACE(species           = [:Hf, :O],
            body_order        = 3,
            polynomial_degree = 3,
            rcutoff           = 5.0,
            wL                = 1.0,
            csp               = 1.0,
            r0                = 1.0)
@save_var path basis

# Update training dataset by adding energy and force descriptors
println("Computing energy descriptors of training dataset...")
B_time = @elapsed e_descr_train = compute_local_descriptors(conf_train, basis)
println("Computing force descriptors of training dataset...")
dB_time = @elapsed f_descr_train = compute_force_descriptors(conf_train, basis)
GC.gc()
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train)

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, basis)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test, basis)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)


# Subsampling experiments: subsample full dataset vs subsample dataset by chunks 

batch_size_prop = 0.1 # subsample 10% of the dataset
n_chunks = 4
n_experiments = 15

metrics_full = []
metrics_split = []
for j in 1:n_experiments
    global metrics_full, metrics_split
    
    # Randomize dataset
    train_ind = randperm(length(ds_train[1:end-1])) # removing off-by-one issue (TODO: improve this)
    ds_train_rnd = @views ds_train[train_ind]

    # Experiment 1: subsample full dataset #####################################
    path_full = "$path/full-trainig-dataset/HfO2-ACE-$j/"
    run(`mkdir -p $path_full`)
    bs = floor(Int, n_train * batch_size_prop)
    dataset_selector = kDPP(  ds_train_rnd,
                              GlobalMean(),
                              DotProduct();
                              batch_size = bs)
    inds_full = get_random_subset(dataset_selector)
    metrics_full_j = fit(path_full, (@views ds_train_rnd[inds_full]), ds_test, basis)
    push!(metrics_full, metrics_full_j)
    
    # Experiment 2: subsample dataset by chunks and then merge #################
    path_split = "$path/split-trainig-dataset/HfO2-ACE-$j/"
    run(`mkdir -p $path_split`)
    inds_split = Int[]
    n_chunk = n_train ÷ n_chunks
    bs = floor(Int, n_chunk * batch_size_prop)
    for i in 1:n_chunks
        a, b = 1 + (i-1) * n_chunk, i * n_chunk
        dataset_selector = kDPP(  ds_train_rnd[a:b],
                                  GlobalMean(),
                                  DotProduct();
                                  batch_size = bs)
        inds_split_i = get_random_subset(dataset_selector)
        append!(inds_split, inds_split_i .+ (a .- 1))
    end
    metrics_split_j = fit(path_split, (@views ds_train_rnd[inds_split]), ds_test, basis)
    push!(metrics_split, metrics_split_j)
end

# Postprocess ##################################################################
metrics = DataFrame( "exp_number" => Int64[],
                     "exp_type"   => String[],
                     "e_train_mae" => Float64[],
                     "e_train_rmse" => Float64[],
                     "e_train_rsq" => Float64[],
                     "f_train_mae" => Float64[],
                     "f_train_rmse" => Float64[],
                     "f_train_rsq" => Float64[],
                     "f_train_mean_cos" => Float64[],
                     "e_test_mae" => Float64[],
                     "e_test_rmse" => Float64[],
                     "e_test_rsq" => Float64[],
                     "f_test_mae" => Float64[],
                     "f_test_rmse" => Float64[],
                     "f_test_rsq" => Float64[],
                     "f_test_mean_cos" => Float64[])
for j in 1:n_experiments
    d1 = merge(OrderedDict("exp_number" => j, "exp_type" => "full"),
               merge(metrics_full[j]...))
    d2 = merge(OrderedDict("exp_number" => j, "exp_type" => "split"),
               merge(metrics_split[j]...))
    push!(metrics, d1)
    push!(metrics, d2)
end
@save_dataframe(path, metrics)

