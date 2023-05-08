# Run this script:
#   $ julia --project=./ --threads=4
#   julia> include("fit-neural-ace.jl")

using AtomsBase
using Unitful, UnitfulAtomic
using InteratomicPotentials 
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra
using Flux
using Optimization
using OptimizationOptimJL
using Random
using BenchmarkTools
include("utils/utils.jl")

# Load input parameters
args = ["experiment_path",      "a-Hfo2-300K-NVT-6000-NeuralACE/",
        "dataset_path",         "data/",
        "dataset_filename",     "a-Hfo2-300K-NVT-6000.extxyz",
        "energy_units",         "eV",
        "distance_units",       "Å",
        "random_seed",          "100",
        "n_train_sys",          "100",
        "n_test_sys",           "100",
        "n_red_desc",           "0", # No. of reduced descriptors. O: don't apply reduction
        "nn",                   "Chain(Dense(n_desc,8,relu),Dense(8,1))",
        "n_epochs",             "100",
        "n_batches",            "1",
        "optimiser",            "Adam(0.001)", # e.g. Adam(0.01) or BFGS()
        "n_body",               "3",
        "max_deg",              "3",
        "r0",                   "1.0",
        "rcutoff",              "5.0",
        "wL",                   "1.0",
        "csp",                  "1.0",
        "w_e",                  "0.01",
        "w_f",                  "1.0",
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
learn_time = @elapsed begin #learn_time = 0.0

# Define ACE parameters
ace = ACE(species = unique(atomic_symbol(get_system(ds[1]))),
          body_order = input["n_body"],
          polynomial_degree = input["max_deg"],
          wL = input["wL"],
          csp = input["csp"],
          r0 = input["r0"],
          rcutoff = input["rcutoff"])

@savevar path ace

# Update training dataset by adding energy and force descriptors
println("Computing energy descriptors of training dataset...")
B_time = @elapsed e_descr_train = compute_local_descriptors(conf_train, ace, T = Float32)
println("Computing force descriptors of training dataset...")
dB_time = @elapsed f_descr_train = compute_force_descriptors(conf_train, ace, T = Float32)
GC.gc()
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train) |> _device
n_desc = length(e_descr_train[1][1]) |> _device

# Dimension reduction of energy and force descriptors of training dataset
reduce_descriptors = input["n_red_desc"] > 0
if reduce_descriptors
    n_desc = input["n_red_desc"]
    pca = PCAState(tol = n_desc)
    fit!(ds_train, pca)
    transform!(ds_train, pca)
end

# Define neural network model
nn = eval(Meta.parse(input["nn"])) # e.g. Chain(Dense(n_desc,8,Flux.leakyrelu), Dense(8,1))
nace = NNIAP(nn, ace)

# Learn
println("Learning energies and forces...")
w_e, w_f = input["w_e"], input["w_f"]
w_e = Float32(w_e)
w_f = Float32(w_f)
opt = eval(Meta.parse(input["optimiser"]))
n_epochs = input["n_epochs"]

# learn!(nace |> cpu, ds_train |> cpu, opt |> cpu, n_epochs, loss, w_e, w_f)
# learn!(nace, ds_train, opt, n_epochs, loss, w_e, w_f, cpu)
n_batches = 100
# learn!(nace, ds_train |> _device, opt |> _device, n_epochs, loss, w_e, w_f, 1.0, 1.0, _device, n_batches)
# learn!(nn |> _device, ace |> _device, ds_train |> _device, opt |> _device, n_epochs, loss, w_e, w_f, 1.0, 1.0, _device, n_batches)


learn!(nace, ds_train, opt, n_epochs, n_batches, loss, w_e, w_f, _device)


benchmark_result = @benchmark learn!(nace, ds_train, opt, 3, n_batches, loss, w_e, w_f, _device)

# Extract the execution times from the benchmark results
execution_times = map(t -> t.time, benchmark_result)

# Create a histogram of the execution times
histogram(execution_times, xlabel="Execution time (ns)", ylabel="Frequency", label="my_function")

end # end of "learn_time = @elapsed begin"



@savevar path Flux.params(nace.nn)

# Post-process output: calculate metrics, create plots, and save results

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, ace, T = Float32)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test, ace, T = Float32)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

# Dimension reduction of energy and force descriptors of test dataset
if reduce_descriptors
    transform!(ds_test, pca)
end

# Get true and predicted values
e_train, f_train = get_all_energies(ds_train), get_all_forces(ds_train)
e_test, f_test = get_all_energies(ds_test), get_all_forces(ds_test)
e_train_pred, f_train_pred = get_all_energies(ds_train, nace), get_all_forces(ds_train, nace)
e_test_pred, f_test_pred = get_all_energies(ds_test, nace), get_all_forces(ds_test, nace)
@savevar path e_train
@savevar path e_train_pred
@savevar path f_train
@savevar path f_train_pred
@savevar path e_test
@savevar path e_test_pred
@savevar path f_test
@savevar path f_test_pred

# Compute metrics
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

