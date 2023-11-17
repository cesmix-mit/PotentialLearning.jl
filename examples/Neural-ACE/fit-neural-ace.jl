# Run this script:
#   $ cd examples/Neural-ACE
#   $ julia --project=../ --threads=4
#   julia> include("fit-neural-ace.jl")

push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials, InteratomicBasisPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using Random
include("../utils/utils.jl")
include("../PCA-ACE/pca.jl")


# Setup experiment #############################################################

# Experiment folder
path = "HfO2-NeuralACE/"
run(`mkdir -p $path/`)

# Fix random seed
Random.seed!(100)


# Define training and test configuration datasets ##############################

ds_path = "../data/HfO2/"

# Load complete configuration dataset
ds_train_path = "$(ds_path)/train/HfO2_mp352_ads_form_sorted.extxyz"
conf_train = load_data(ds_train_path, uparse("eV"), uparse("Å"))

ds_test_path = "$(ds_path)/test/Hf_mp103_ads_form_sorted.extxyz"
conf_test = load_data(ds_test_path, uparse("eV"), uparse("Å"))

n_train, n_test = length(conf_train), length(conf_test)

species = unique(vcat([atomic_symbol.(get_system(c).particles)
          for c in conf_train]...))

# Define dataset subselector ###################################################

# Subselector, option 1: RandomSelector
dataset_selector = RandomSelector(length(conf_train); batch_size = 76)

# Subselector, option 2: DBSCANSelector. Pre-cond: const. no. of atoms
#ε, min_pts, sample_size = 0.05, 5, 3
#dataset_selector = DBSCANSelector(  conf_train,
#                                    ε,
#                                    min_pts,
#                                    sample_size)

# Subselector, option 3: kDPP + ACE (requires calculation of energy descriptors)
#ace = ACE(species           = [:Hf, :O],
#          body_order        = 2,
#          polynomial_degree = 3,
#          wL                = 1.0,
#          csp               = 1.0,
#          r0                = 1.0,
#          rcutoff           = 5.0)
#e_descr = compute_local_descriptors(conf_train,
#                                    ace,
#                                    T = Float32)
#conf_train_kDPP = DataSet(conf_train .+ e_descr)
#dataset_selector = kDPP(  conf_train_kDPP,
#                          GlobalMean(),
#                          DotProduct();
#                          batch_size = 75)

# Subsample trainig dataset
inds = PotentialLearning.get_random_subset(dataset_selector)
conf_train = conf_train[inds]
GC.gc()


# Define IAP model #############################################################

# Define ACE
ace = ACE(species           = [:Hf, :O],
          body_order        = 3,
          polynomial_degree = 3,
          wL                = 1.0,
          csp               = 1.0,
          r0                = 1.0,
          rcutoff           = 5.0)
@save_var path ace

# Update training dataset by adding energy descriptors
println("Computing energy descriptors of training dataset...")
e_descr_train = compute_local_descriptors(conf_train,
                                          ace,
                                          T = Float32)
ds_train = DataSet(conf_train .+ e_descr_train)


# Dimension reduction of energy and force descriptors of training dataset ######
reduce_descriptors = false
n_desc = length(e_descr_train[1][1])
if reduce_descriptors
    n_desc = n_desc / 2
    pca = PCAState(tol = n_desc)
    fit!(ds_train, pca)
    transform!(ds_train, pca)
end

# Define neural network model
nns = Dict()
for s in species
    nns[s] = Chain( Dense(n_desc,128,σ; init = Flux.glorot_uniform(gain=-10)),
                    Dense(128,128,σ; init = Flux.glorot_uniform(gain=-10)),
                    Dense(128,1; init = Flux.glorot_uniform(gain=-10), bias = false))
end
nace = NNIAP(nns, ace)

# Learn
println("Learning energies...")

opt = Adam(1e-2)
n_epochs = 50
log_step = 10
batch_size = 4
w_e, w_f = 1.0, 0.0
reg = 1e-8
learn!(nace,
       ds_train,
       opt,
       n_epochs,
       energy_loss,
       w_e,
       w_f,
       reg,
       batch_size,
       log_step
)

opt = Adam(1e-4)
n_epochs = 500
log_step = 10
batch_size = 4
w_e, w_f = 1.0, 0.0
reg = 1e-4
learn!(nace,
       ds_train,
       opt,
       n_epochs,
       energy_loss,
       w_e,
       w_f,
       reg,
       batch_size,
       log_step
)

# Save current NN parameters
ps1, _ = Flux.destructure(nace.nns[:Hf])
ps2, _ = Flux.destructure(nace.nns[:O])
@save_var path ps1
@save_var path ps2


# Post-process output: calculate metrics, create plots, and save results #######

# Dimension reduction of energy and force descriptors of test dataset
if reduce_descriptors
    transform!(ds_test, pca)
end

# Update test dataset by adding energy descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test,
                                         ace,
                                         T = Float32)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test)

# Get true and predicted values
n_atoms_train = length.(get_system.(ds_train))
n_atoms_test = length.(get_system.(ds_test))

e_train, e_train_pred = get_all_energies(ds_train) ./ n_atoms_train,
                        get_all_energies(ds_train, nace) ./ n_atoms_train
@save_var path e_train
@save_var path e_train_pred

e_test, e_test_pred = get_all_energies(ds_test) ./ n_atoms_test,
                      get_all_energies(ds_test, nace) ./ n_atoms_test
@save_var path e_test
@save_var path e_test_pred

# Compute metrics
e_train_metrics = get_metrics(e_train_pred,
                              e_train,
                              metrics = [mae, rmse, rsq],
                              label = "e_train")
@save_dict path e_train_metrics

e_test_metrics = get_metrics(e_test_pred,
                             e_test,
                             metrics = [mae, rmse, rsq],
                             label = "e_test")
@save_dict path e_test_metrics

# Plot and save results
e_train_plot = plot_energy(e_train_pred, e_train)
@save_fig path e_train_plot

e_test_plot = plot_energy(e_test_pred, e_test)
@save_fig path e_test_plot

e_plot = plot_energy(e_train_pred, e_train,
                     e_test_pred, e_test)
@save_fig path e_plot

