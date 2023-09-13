# Run this script:
#   $ cd examples/HyperLearn
#   $ julia --project=../ --threads=4
#   julia> include("fit-opt-ace.jl")

push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials, InteratomicBasisPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using Random
include("../utils/utils.jl")
include("HyperLearn.jl")


# Load input parameters
args = ["experiment_path",      "a-HfO2-Opt-ACE/",
        "dataset_path",         "../data/a-HfO2/",
        "dataset_filename",     "a-Hfo2-300K-NVT-6000.extxyz",
        "energy_units",         "eV",
        "distance_units",       "Å",
        "random_seed",          "100",
        "n_train_sys",          "500",
        "n_test_sys",           "500",
        "e_mae_tol",            "0.2",
        "f_mae_tol",            "0.2",
        "sample_size",          "10",
        "eps",                  "0.05",
        "minpts",               "5",
        "n_body",               "3",
        "max_deg",              "3",
        "r0",                   "1.0",
        "rcutoff",              "5.0",
        "wL",                   "1.0",
        "csp",                  "1.0",
        "w_e",                  "1.0",
        "w_f",                  "1.0"]
args = length(ARGS) > 0 ? ARGS : args
input = get_input(args)


# Create experiment folder
path = input["experiment_path"]
run(`mkdir -p $path`)
@savecsv path input

# Fix random seed
if "random_seed" in keys(input)
    Random.seed!(input["random_seed"])
end

# Load dataset
ds_path = input["dataset_path"]*input["dataset_filename"] # dirname(@__DIR__)*"/data/"*input["dataset_filename"]
energy_units, distance_units = uparse(input["energy_units"]), uparse(input["distance_units"])
ds = load_data(ds_path, energy_units, distance_units)

# Configuration datasets
n_train, n_test = input["n_train_sys"], input["n_test_sys"]
conf_train, conf_test = split(ds, n_train, n_test)

# Dataset selector
epsi, minpts, sample_size = input["eps"], input["minpts"], input["sample_size"]
dataset_selector = DBSCANSelector(  conf_train,
                                    epsi,
                                    minpts,
                                    sample_size) 
#ss = kDPP(conf_train, GlobalMean(), DotProduct(); batch_size = 200)

# Dataset generator
dataset_generator = Nothing

# IAP model
model = ACE

# IAP parameter subspace
pars = OrderedDict(
        :species           => [[:Hf, :O]],
        :body_order        => [2, 3],
        :polynomial_degree => [3, 4],
        :wL                => [1],
        :csp               => [1],
        :r0                => [1],
        :rcutoff           => [5, 5.5])

# Hyperoptimizer
n_samples = 10
hyper_optimizer = Hyperoptimizer(n_samples,
                                 RandomSampler();
                                 pars...)

# Maximum no. of iterations
max_iterations = 1

# End condition
end_condition() = return false

# Weights
weights = [input["w_e"], input["w_f"]]

# Hyper-learn
hyperlearn!(hyper_optimizer,
            model,
            conf_train,
            dataset_selector,
            dataset_generator,
            max_iterations,
            end_condition,
            weights)

# Optimal IAP
opt_iap = get_opt_iap(hyper_optimizer, model)

# Show optimization results
@show sortslices(hcat(hyper_optimizer.results, hyper_optimizer.history),
                 dims = 1,
                 by = x->x[1])
best_params, min_f = hyper_optimizer.minimizer, hyper_optimizer.minimum

@savevar path opt_iap.β

# Post-process output: calculate metrics, create plots, and save results

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, opt_iap.basis)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test, opt_iap.basis)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

# Get true and predicted values
e_test, f_test =    get_all_energies(ds_test),
                    get_all_forces(ds_test)
e_test_pred, f_test_pred =    get_all_energies(ds_test, opt_iap),
                              get_all_forces(ds_test, opt_iap)
@savevar path e_test
@savevar path e_test_pred
@savevar path f_test
@savevar path f_test_pred

# Compute metrics
metrics = get_metrics(e_test_pred, e_test,
                      f_test_pred, f_test)
@savecsv path metrics

# Plot and save results
e_test_plot = plot_energy(e_test_pred, e_test)
@savefig path e_test_plot
f_test_plot = plot_forces(f_test_pred, f_test)
@savefig path f_test_plot
f_test_cos = plot_cos(f_test_pred, f_test)
@savefig path f_test_cos

