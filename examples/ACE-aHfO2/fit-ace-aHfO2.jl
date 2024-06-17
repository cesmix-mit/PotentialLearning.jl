# # Fit a-HfO2 dataset with ACE

# ## Load packages, define paths, and create experiment folder.

# Load packages
using AtomsBase, InteratomicPotentials, PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra, Random, DisplayAs

# Define paths.
path = joinpath(dirname(pathof(PotentialLearning)), "../examples/ACE-aHfO2")
ds_path =  "$path/../data/a-HfO2/a-HfO2-300K-NVT-6000.extxyz"
res_path = "$path/results/"

# Load utility functions.
include("$path/../utils/utils.jl")

# Create experiment folder.
run(`mkdir -p $res_path`)

# ## Load atomistic dataset and split it into training and test.

# Load atomistic dataset: atomistic configurations (atom positions, geometry, etc.) + DFT data (energies, forces, etc.)
ds = load_data(ds_path, uparse("eV"), uparse("Å"))

# Split atomistic dataset into training and test
n_train, n_test = 50, 50 # only 50 samples per dataset are used in this example.
conf_train, conf_test = split(ds[1:1000], n_train, n_test)

# ## Create ACE basis, compute descriptors and add them to the dataset.

# Create ACE basis
basis = ACE(species           = [:Hf, :O],
            body_order        = 3,
            polynomial_degree = 4,
            rcutoff           = 5.0,
            wL                = 1.0,
            csp               = 1.0,
            r0                = 1.0)
@save_var res_path basis

# Compute ACE descriptors for energy and forces based on the atomistic training configurations.
println("Computing energy descriptors of training dataset...")
e_descr_train = compute_local_descriptors(conf_train, basis;
                                          pbar=false)
println("Computing force descriptors of training dataset...")
f_descr_train = compute_force_descriptors(conf_train, basis;
                                          pbar=false)

# Update training dataset by adding energy and force descriptors.
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train)

# ## Learn ACE coefficients based on ACE descriptors and DFT data.
println("Learning energies and forces...")
lb = LBasisPotential(basis)
ws, int = [1.0, 1.0], false
learn!(lb, ds_train, ws, int)
@save_var res_path lb.β
@save_var res_path lb.β0
lb.β, lb.β0

# ## Post-process output: calculate metrics, create plots, and save results.

# Compute ACE descriptors for energy and forces based on the atomistic test configurations.
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, basis;
                                         pbar = false)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test, basis;
                                         pbar = false)

# Update test dataset by adding energy and force descriptors.
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

# Get true and predicted values for energies and forces.
n_atoms_train = length.(get_system.(ds_train))
n_atoms_test = length.(get_system.(ds_test))

e_train, e_train_pred = get_all_energies(ds_train) ./ n_atoms_train,
                        get_all_energies(ds_train, lb) ./ n_atoms_train
f_train, f_train_pred = get_all_forces(ds_train),
                        get_all_forces(ds_train, lb)
@save_var res_path e_train
@save_var res_path e_train_pred
@save_var res_path f_train
@save_var res_path f_train_pred

e_test, e_test_pred = get_all_energies(ds_test) ./ n_atoms_test,
                      get_all_energies(ds_test, lb) ./ n_atoms_test
f_test, f_test_pred = get_all_forces(ds_test),
                      get_all_forces(ds_test, lb)
@save_var res_path e_test
@save_var res_path e_test_pred
@save_var res_path f_test
@save_var res_path f_test_pred

# Compute training metrics
e_train_metrics = get_metrics(e_train, e_train_pred,
                              metrics = [mae, rmse, rsq],
                              label = "e_train")
f_train_metrics = get_metrics(f_train, f_train_pred,
                              metrics = [mae, rmse, rsq, mean_cos],
                              label = "f_train")
train_metrics = merge(e_train_metrics, f_train_metrics)
@save_dict res_path train_metrics
train_metrics

# Compute test metrics
e_test_metrics = get_metrics(e_test, e_test_pred,
                             metrics = [mae, rmse, rsq],
                             label = "e_test")
f_test_metrics = get_metrics(f_test, f_test_pred,
                             metrics = [mae, rmse, rsq, mean_cos],
                             label = "f_test")
test_metrics = merge(e_test_metrics, f_test_metrics)
@save_dict res_path test_metrics
test_metrics

# Plot and save energy results
e_plot = plot_energy(e_train, e_train_pred,
                     e_test, e_test_pred)
@save_fig res_path e_plot
DisplayAs.PNG(e_plot)

# Plot and save force results
f_plot = plot_forces(f_train, f_train_pred,
                     f_test, f_test_pred)
@save_fig res_path f_plot
DisplayAs.PNG(f_plot)

# Plot and save training force cosine
e_train_plot = plot_energy(e_train, e_train_pred)
f_train_plot = plot_forces(f_train, f_train_pred)
f_train_cos  = plot_cos(f_train, f_train_pred)
@save_fig res_path e_train_plot
@save_fig res_path f_train_plot
@save_fig res_path f_train_cos
DisplayAs.PNG(f_train_cos)

# Plot and save test force cosine
e_test_plot = plot_energy(e_test, e_test_pred)
f_test_plot = plot_forces(f_test, f_test_pred)
f_test_cos  = plot_cos(f_test, f_test_pred)
@save_fig res_path e_test_plot
@save_fig res_path f_test_plot
@save_fig res_path f_test_cos
DisplayAs.PNG(f_test_cos)

