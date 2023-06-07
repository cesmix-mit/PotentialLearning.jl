# Run this script:
#   $ cd examples/ACE
#   $ julia --project=../ --threads=4
#   julia> include("fit-pace-ace-hyperopt.jl")

push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials, InteratomicBasisPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using Hyperopt
using Random
include("../utils/utils.jl")
include("PCAL.jl")


# Load input parameters
args = ["experiment_path",      "a-HfO2-PCAL-ACE-HYPEROPT/",
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
        "eps",                  "0.5",
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

# Start hyperopt loop
ho = @thyperopt for i = 18,
                   body_order_h = [2,3],
                   polynomial_degree_h = [3,4],
                   rcutoff_h = [5]

# Define ACE
ace = ACE(species = unique(atomic_symbol(get_system(ds[1]))),
          body_order = body_order_h,
          polynomial_degree = polynomial_degree_h,
          wL = input["wL"],
          csp = input["csp"],
          r0 = input["r0"],
          rcutoff = rcutoff_h)
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
e_mae, e_rmse, e_rsq, f_mae, f_rmse, f_rsq = learn2(pcal, ds_train, 5)

f_mae # return value of hyperopt loop (we are optimizing for f_mae)
end # end hyperopt

@show sortslices(hcat(ho.results, ho.history), dims=1, by=x->x[1])
