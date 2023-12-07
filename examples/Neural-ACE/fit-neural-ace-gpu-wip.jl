# Run this script:
#   $ cd examples/Neural-ACE
#   $ julia --project=../ --threads=4
#   julia> include("fit-neural-ace-gpu-wip.jl")

push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using Random
include("../utils/utils.jl")

# Load input parameters
args = ["experiment_path",      "a-Hfo2-300K-NVT-6000-NeuralACE-GPU-WIP/",
        "dataset_path",         "../data/a-HfO2/",
        "dataset_filename",     "a-Hfo2-300K-NVT-6000.extxyz",
        "energy_units",         "eV",
        "distance_units",       "Å",
        "random_seed",          "100",
        "n_train_sys",          "100",
        "n_test_sys",           "100",
        "nn",                   "Chain(Dense(n_desc,8,relu),Dense(8,1))",
#        "n_epochs",             "10000",
#        "n_batches",            "1",
#        "optimiser",            "Adam(0.001, (.9, .8))", # e.g. Adam(0.01) or BFGS()
        "n_body",               "3",
        "max_deg",              "3",
        "r0",                   "1.0",
        "rcutoff",              "5.0",
        "wL",                   "1.0",
        "csp",                  "1.0",
        "w_e",                  "1.0",
        "device",              "gpu"]

args = length(ARGS) > 0 ? ARGS : args
input = get_input(args)


# Create experiment folder
path = input["experiment_path"]
run(`mkdir -p $path`)
@savecsv path input

#Get device
_device = input["device"]
_device = _device == "gpu" ? gpu : cpu

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

# Define ACE parameters
ace = ACE(species = unique(atomic_symbol(get_system(ds[1]))),
          body_order = input["n_body"],
          polynomial_degree = input["max_deg"],
          wL = input["wL"],
          csp = input["csp"],
          r0 = input["r0"],
          rcutoff = input["rcutoff"])

@savevar path ace

# Update training dataset by adding energy descriptors
println("Computing energy descriptors of training dataset...")
B_time = @elapsed e_descr_train = compute_local_descriptors(conf_train, ace, T = Float32)
ds_train = DataSet(conf_train .+ e_descr_train)
n_desc = length(e_descr_train[1][1])

# Define neural network model
nn = eval(Meta.parse(input["nn"])) # e.g. Chain(Dense(n_desc,8,Flux.leakyrelu), Dense(8,1))
nace = NNIAP(nn, ace)

# Learn
println("Learning energies...")
w_e, w_f = input["w_e"], 0.0
#opt = eval(Meta.parse(input["optimiser"]))
#n_epochs = input["n_epochs"]
#n_batches = input["n_batches"]
#learn!(nace, ds_train, opt, n_epochs, loss, w_e, w_f)

n_batches = 1

n_epochs = 200
opt = Adam(0.1, (.9, .8))
learn!(nace, ds_train, opt, n_epochs, n_batches, loss, w_e, w_f, _device)

opt = Adam(0.001, (.9, .8))
n_epochs = 200_000
learn!(nace, ds_train, opt, n_epochs, n_batches, loss, w_e, w_f, _device)

end

@savevar path Flux.params(nace.nn)

# Post-process output: calculate metrics, create plots, and save results

# Update test dataset by adding energy descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, ace, T = Float32)
ds_test = DataSet(conf_test .+ e_descr_test)

# Get true and predicted values
e_train, e_train_pred = get_all_energies(ds_train), get_all_energies(ds_train, nace)
e_test, e_test_pred = get_all_energies(ds_test), get_all_energies(ds_test, nace)
@savevar path e_train
@savevar path e_train_pred
@savevar path e_test
@savevar path e_test_pred

# Compute metrics
metrics = get_metrics(e_train_pred, e_train, e_test_pred, e_test)
@savecsv path metrics

# Plot and save results
e_test_plot = plot_energy(e_test_pred, e_test)
@savefig path e_test_plot


