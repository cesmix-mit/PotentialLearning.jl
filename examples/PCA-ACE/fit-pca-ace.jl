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
include("../utils/utils.jl")
include("../PCA-ACE/pca.jl")


# Setup experiment #############################################################

# Experiment folder
path = "a-HfO2-PCA-ACE/"
run(`mkdir -p $path`)

# Define training and test configuration datasets ##############################

# Load complete configuration dataset
ds_path = string("../data/a-HfO2/a-Hfo2-300K-NVT-6000.extxyz")
ds = load_data(ds_path, uparse("eV"), uparse("Å"))

# Split configuration dataset into training and test
n_train, n_test = 100, 100
conf_train, conf_test = split(ds, n_train, n_test)

# Define dataset subselector ###################################################

# Subselector, option 1: RandomSelector
#dataset_selector = RandomSelector(length(conf_train); batch_size = 100)

# Subselector, option 2: DBSCANSelector
#ε, min_pts, sample_size = 0.05, 5, 3
#dataset_selector = DBSCANSelector(  conf_train,
#                                    ε,
#                                    min_pts,
#                                    sample_size)

# Subselector, option 3: kDPP + ACE (requires calculation of energy descriptors)
#basis = ACE(species           = [:Hf, :O],
#            body_order        = 2,
#            polynomial_degree = 3,
#            wL                = 1.0,
#            csp               = 1.0,
#            r0                = 1.0,
#            rcutoff           = 5.0)
#e_descr = compute_local_descriptors(conf_train,
#                                    basis,
#                                    pbar = false)
#conf_train_kDPP = DataSet(conf_train .+ e_descr)
#dataset_selector = kDPP(  conf_train_kDPP,
#                          GlobalMean(),
#                          DotProduct();
#                          batch_size = 100)

# Subsample trainig dataset
#inds = PotentialLearning.get_random_subset(dataset_selector)
#conf_train = conf_train[inds]
#GC.gc()

# Define IAP model #############################################################

# Define ACE
basis = ACE(species           = [:Hf, :O],
            body_order        = 3,
            polynomial_degree = 4,
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
n_desc = length(e_descr_train[1][1])

# Dimension reduction of energy and force descriptors of training dataset
reduce_descriptors = true
if reduce_descriptors
    n_desc = 20
    pca = PCAState(tol = n_desc)
    fit!(ds_train, pca)
    transform!(ds_train, pca)
end

# Learn
println("Learning energies and forces...")
lb = LBasisPotential(basis)
ws, int = [1.0, 1.0], true
learn!(lb, ds_train, ws, int)

@save_var path lb.β
@save_var path lb.β0

# Post-process output: calculate metrics, create plots, and save results #######

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, basis)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test, basis)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

# Dimension reduction of energy and force descriptors of test dataset
if reduce_descriptors
    transform!(ds_test, pca)
end

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

