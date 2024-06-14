using AtomsBase, InteratomicPotentials, PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra, Random
using DisplayAs

path = joinpath(dirname(pathof(PotentialLearning)), "../examples/ACE-aHfO2")

include("$path/../utils/utils.jl")

# Setup experiment

# Experiment folder
path = "$path/results/"
run(`mkdir -p $path`)

# Define training and test configuration datasets

# Load complete configuration dataset
ds_path = string("$path/../../data/a-HfO2/a-HfO2-300K-NVT-6000.extxyz")
ds = load_data(ds_path, uparse("eV"), uparse("Å"))

# Split configuration dataset into training and test
n_train, n_test = 50, 50
conf_train, conf_test = split(ds, n_train, n_test)

# Define IAP model

# Define ACE basis
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

# Learn
println("Learning energies and forces...")
lb = LBasisPotential(basis)
ws, int = [1.0, 1.0], false
learn!(lb, ds_train, ws, int)

@save_var path lb.β
@save_var path lb.β0
lb.β, lb.β0

# Post-process output: calculate metrics, create plots, and save results

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, basis)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test, basis)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

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

# Compute training metrics
e_train_metrics = get_metrics(e_train, e_train_pred,
                              metrics = [mae, rmse, rsq],
                              label = "e_train")
f_train_metrics = get_metrics(f_train, f_train_pred,
                              metrics = [mae, rmse, rsq, mean_cos],
                              label = "f_train")
train_metrics = merge(e_train_metrics, f_train_metrics)
@save_dict path train_metrics
train_metrics

# Compute test metrics
e_test_metrics = get_metrics(e_test, e_test_pred,
                             metrics = [mae, rmse, rsq],
                             label = "e_test")
f_test_metrics = get_metrics(f_test, f_test_pred,
                             metrics = [mae, rmse, rsq, mean_cos],
                             label = "f_test")
test_metrics = merge(e_test_metrics, f_test_metrics)
@save_dict path test_metrics
test_metrics

# Plot and save energy results
e_plot = plot_energy(e_train, e_train_pred,
                     e_test, e_test_pred)
@save_fig path e_plot
DisplayAs.PNG(e_plot)

# Plot and save force results
f_plot = plot_forces(f_train, f_train_pred,
                     f_test, f_test_pred)
@save_fig path f_plot
DisplayAs.PNG(f_plot)

# Plot and save training force cosine
e_train_plot = plot_energy(e_train, e_train_pred)
f_train_plot = plot_forces(f_train, f_train_pred)
f_train_cos  = plot_cos(f_train, f_train_pred)
@save_fig path e_train_plot
@save_fig path f_train_plot
@save_fig path f_train_cos
DisplayAs.PNG(f_train_cos)

# Plot and save test force cosine
e_test_plot = plot_energy(e_test, e_test_pred)
f_test_plot = plot_forces(f_test, f_test_pred)
f_test_cos  = plot_cos(f_test, f_test_pred)
@save_fig path e_test_plot
@save_fig path f_test_plot
@save_fig path f_test_cos
DisplayAs.PNG(f_test_cos)

