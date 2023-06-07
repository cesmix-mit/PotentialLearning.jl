# Run this script:
#   $ cd examples/ACE
#   $ julia --project=../ --threads=4
#   julia> include("fit-pace-ace.jl")

push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials, InteratomicBasisPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using Random
include("../utils/utils.jl")
include("PCAL.jl")


# Load input parameters
args = ["experiment_path",      "a-HfO2-PCAL-ACE/",
        "dataset_path",         "../data/a-HfO2/",
        "dataset_filename",     "a-Hfo2-300K-NVT-6000.extxyz",
        "energy_units",         "eV",
        "distance_units",       "Å",
        "random_seed",          "100",
        "n_train_sys",          "800",
        "n_test_sys",           "200",
        "e_mae_tol",            "0.2",
        "f_mae_tol",            "0.2",
        "sample_size",          "10",
        "eps",                  "0.05",
        "minpts",               "10",
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

# Split dataset
n_train, n_test = input["n_train_sys"], input["n_test_sys"]
conf_train, conf_test = split(ds, n_train, n_test)

# Start measuring learning time
learn_time = @elapsed begin

# Define ACE
ace = ACE(species = unique(atomic_symbol(get_system(ds[1]))),
          body_order = input["n_body"],
          polynomial_degree = input["max_deg"],
          wL = input["wL"],
          csp = input["csp"],
          r0 = input["r0"],
          rcutoff = input["rcutoff"])
@savevar path ace

# Learn
println("Learning energies and forces...")
ds_train = conf_train
lb = LBasisPotential(ace)
pcal = PCALProblem(lb;
                   e_mae_tol = input["e_mae_tol"],
                   f_mae_tol = input["f_mae_tol"],
                   sample_size = input["sample_size"],
                   eps = input["eps"],
                   minpts = input["minpts"],
                   w_e = input["w_e"],
                   w_f = input["w_f"])
learn!(pcal, ds_train)

end # end of "learn_time = @elapsed begin"

@savevar path lb.β

# Post-process output: calculate metrics, create plots, and save results

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, ace)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test, ace)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)


# Get true and predicted values
e_train, f_train = get_all_energies(ds_train), get_all_forces(ds_train)
e_test, f_test = get_all_energies(ds_test), get_all_forces(ds_test)
e_train_pred, f_train_pred = get_all_energies(ds_train, lb), get_all_forces(ds_train, lb)
e_test_pred, f_test_pred = get_all_energies(ds_test, lb), get_all_forces(ds_test, lb)
@savevar path e_train
@savevar path e_train_pred
@savevar path f_train
@savevar path f_train_pred
@savevar path e_test
@savevar path e_test_pred
@savevar path f_test
@savevar path f_test_pred

# Compute metrics
B_time = dB_time = 0.0
metrics = get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                       e_test_pred, e_test, f_test_pred, f_test,
                       B_time, dB_time, learn_time)
@savecsv path metrics

# Plot and save results
e_test_plot = plot_energy(e_test_pred, e_test)
@savefig path e_test_plot
f_test_plot = plot_forces(f_test_pred, f_test)
@savefig path f_test_plot
f_test_cos = plot_cos(f_test_pred, f_test)
@savefig path f_test_cos

