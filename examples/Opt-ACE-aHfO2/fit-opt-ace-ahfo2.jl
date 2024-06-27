# # Optimize ACE hyper-parameters: minimize force time and fitting error.

# ## a. Load packages, define paths, and create experiment folder.

# Load packages.
using AtomsBase, InteratomicPotentials, PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra, Random, DisplayAs
using DataFrames, Hyperopt

# Define paths.
path = joinpath(dirname(pathof(PotentialLearning)), "../examples/Opt-ACE-aHfO2")
ds_path =  "$path/../data/a-HfO2/a-HfO2-300K-NVT-6000.extxyz"
res_path = "$path/results/";

# Load utility functions.
include("$path/../utils/utils.jl")
include("$path/utils.jl")

# Create experiment folder.
run(`mkdir -p $res_path`);

# ## b. Load atomistic dataset and split it into training and test.

# Load atomistic dataset: atomistic configurations (atom positions, geometry, etc.) + DFT data (energies, forces, etc.)
ds = load_data(ds_path, uparse("eV"), uparse("Å"))[1:1000]

# Split atomistic dataset into training and test
n_train, n_test = 50, 50 # Only 50 samples per dataset are used in this example.
conf_train, conf_test = split(ds, n_train, n_test)


# ## c. Hyper-parameter optimization.

# Define the model and hyper-parameter value ranges to be optimized.
model = ACE
pars = OrderedDict( :body_order        => [2, 3, 4],
                    :polynomial_degree => [3, 4, 5],
                    :rcutoff           => [4.5, 5.0, 5.5],
                    :wL                => [0.5, 1.0, 1.5],
                    :csp               => [0.5, 1.0, 1.5],
                    :r0                => [0.5, 1.0, 1.5]);
# Use random sampling to find the optimal hyper-parameters.
iap, res = hyperlearn!(model, pars, conf_train; 
                       n_samples = 10, sampler = RandomSampler(),
                       ws = [1.0, 1.0], int = true)

@save_var res_path iap.β
@save_var res_path iap.β0
@save_var res_path iap.basis
@save_dataframe res_path res
res

# Plot error vs time
err_time = plot_err_time(res)
@save_fig res_path err_time
DisplayAs.PNG(err_time)


# Alternatively, use latin hypercube sampling to find the optimal hyper-parameters.
iap, res = hyperlearn!(model, pars, conf_train; 
                       n_samples = 3, sampler = LHSampler(),
                       ws = [1.0, 1.0], int = true)

@save_var res_path iap.β
@save_var res_path iap.β0
@save_var res_path iap.basis
@save_dataframe res_path res
res

# Plot error vs time
err_time = plot_err_time(res)
@save_fig res_path err_time
DisplayAs.PNG(err_time)


