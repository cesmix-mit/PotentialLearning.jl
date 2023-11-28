# Run this script:
#   $ cd examples/ACE
#   $ julia --project=../ --threads=4
#   julia> include("fit-cnn-ace.jl")

push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials, InteratomicBasisPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using Random
include("../utils/utils.jl")
include("PL-IBS-Ext.jl")


# Setup experiment #############################################################

# Experiment folder
path = "aHfO2-CNNACE-b/"
run(`mkdir -p $path/`)

# Fix random seed
Random.seed!(100)

# Define training and test configuration datasets ##############################

ds_path = "../data/a-HfO2/a-Hfo2-300K-NVT-6000.extxyz" 
ds = load_data(ds_path, uparse("eV"), uparse("Å"))
#ds = ds[shuffle(1:length(ds))]

# Split configuration dataset into training and test
n_train, n_test = 3000, 1000
conf_train, conf_test = split(ds, n_train, n_test)

species = unique(vcat([atomic_symbol.(get_system(c).particles)
          for c in conf_train]...))

# Define ACE parameters
ace = ACE(species = unique(atomic_symbol(get_system(ds[1]))),
          body_order = 3,
          polynomial_degree = 3,
          wL = 1.0,
          csp = 1.0,
          r0 = 1.0,
          rcutoff = 5.0)
@save_var path ace

# Update training dataset by adding energy and force descriptors
println("Computing energy descriptors of training dataset...")
e_descr_train = compute_local_descriptors(conf_train,
                                          ace,
                                          T = Float32)
ds_train = DataSet(conf_train .+ e_descr_train)
n_desc = length(e_descr_train[1][1])

# Update test dataset by adding energy descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test,
                                         ace,
                                         T = Float32)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test)


# Define neural network model
n_atoms = length(get_system(first(ds_train)))
n_types = length(ace.species)
n_basis = length(first(get_values(get_local_descriptors(first(ds_train))))) ÷ n_types
batch_size = length(ds_train)

#nn = Flux.@autosize (n_atoms, n_basis, n_types, batch_size) Chain(
#    Conv((3, 3), 2=>6, relu),
#    MaxPool((2, 2)),
#    Conv((3, 3), _=>16, relu),
#    MaxPool((2, 2)),
#    Flux.flatten,
#    Dense(_ => 120, relu),
#    Dense(_ => 84, relu), 
#    Dense(_ => 1),
#)

nns = Flux.@autosize (n_atoms, n_basis, n_types, batch_size) Chain(
    BatchNorm(_),
    Conv((1, 3), 2=>16),
    BatchNorm(_, relu),
    MeanPool((1, 2)),
    Conv((1, 3), _=>32),
    BatchNorm(_, relu),
    MeanPool((1, 2)),
    Flux.flatten,
#    Dropout(0.2),
#    Dense(_ => 120, relu),
#    Dense(_ => 84, relu), 
    Dense(_ => 1),
)

#nn = Flux.@autosize (n_types, n_basis, n_atoms, batch_size) Chain(
##    BatchNorm(_, affine=true, relu),
#    Conv((1, 4), n_atoms=>6, relu),
#    MaxPool((1, 2)),
#    Conv((1, 4), _=>16, relu),
#    MaxPool((1, 2)),
#    Flux.flatten,
##    Dropout(0.8),
#    Dense(_ => 120, relu),
#    Dense(_ => 84, relu), 
#    Dense(_ => 1)
#)

cnnace = NNIAP(nns, ace)

# Learn
println("Learning energies and forces...")
η = 1e-4         # learning rate
λ = 1e-2         # for weight decay
opt_rule = OptimiserChain(WeightDecay(λ), Adam(η, (0.9, 0.8)))
opt_state = Flux.setup(opt_rule, nns)
n_epochs = 10_000
learn!(cnnace, ds_train, ds_test, opt_state, n_epochs, loss)
@save_var path Flux.params(cnnace.nns)

η = 1e-6         # learning rate
λ = 1e-3         # for weight decay
opt_rule = OptimiserChain(WeightDecay(λ), Adam(η, (0.9, 0.8)))
opt_state = Flux.setup(opt_rule, nns)
n_epochs = 50_000
learn!(cnnace, ds_train, ds_test, opt_state, n_epochs, loss)
@save_var path Flux.params(cnnace.nns)


# Post-process output: calculate metrics, create plots, and save results #######

# Get true and predicted values
n_atoms_train = length.(get_system.(ds_train))
n_atoms_test = length.(get_system.(ds_test))

e_train, e_train_pred = get_all_energies(ds_train) ./ n_atoms_train,
                        get_all_energies(ds_train, cnnace) ./ n_atoms_train
@save_var path e_train
@save_var path e_train_pred

e_test, e_test_pred = get_all_energies(ds_test) ./ n_atoms_test,
                      get_all_energies(ds_test, cnnace) ./ n_atoms_test
@save_var path e_test
@save_var path e_test_pred

# Compute metrics
e_train_metrics = get_metrics(e_train,
                              e_train_pred,
                              metrics = [mae, rmse, rsq],
                              label = "e_train")
@save_dict path e_train_metrics

e_test_metrics = get_metrics(e_test,
                             e_test_pred,
                             metrics = [mae, rmse, rsq],
                             label = "e_test")
@save_dict path e_test_metrics

# Plot and save results
e_train_plot = plot_energy(e_train, e_train_pred)
@save_fig path e_train_plot

e_test_plot = plot_energy(e_test, e_test_pred)
@save_fig path e_test_plot

e_plot = plot_energy(e_train, e_train_pred,
                     e_test, e_test_pred)
@save_fig path e_plot

