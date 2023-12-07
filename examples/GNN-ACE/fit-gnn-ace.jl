# Run this script:
#   $ cd examples/GNN-ACE
#   $ julia --project=../ --threads=4
#   julia> include("fit-gnn-ace.jl")

push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials, InteratomicBasisPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using GraphNeuralNetworks
using Random
using JLD
include("../utils/utils.jl")
include("gnniap.jl")


# Setup experiment #############################################################

# Experiment folder
path = "Hf-GNNACE/"
#path = "aHfO2-GNNACE/"
run(`mkdir -p $path/`)

# Fix random seed
Random.seed!(100)

# Use cpu or gpu
device = Flux.gpu

# Define training and test configuration datasets ##############################

# Load complete configuration dataset
#ds_path = "../data/HfO2/"
#ds_train_path = "$(ds_path)/train/HfO2_mp352_ads_form_sorted.extxyz"
#conf_train = load_data(ds_train_path, uparse("eV"), uparse("Å"))
#ds_test_path = "$(ds_path)/test/Hf_mp103_ads_form_sorted.extxyz"
#conf_test = load_data(ds_test_path, uparse("eV"), uparse("Å"))
#n_train, n_test = length(conf_train), length(conf_test)

# Load complete configuration dataset
ds_path = "../data/Hf/Hf128_MC_rattled_random_form_sorted.extxyz"
#ds_path = "../data/a-HfO2/a-Hfo2-300K-NVT-6000.extxyz" 
ds = load_data(ds_path, uparse("eV"), uparse("Å"))
ds = ds[shuffle(1:length(ds))]

# Split configuration dataset into training and test
n_train, n_test = 400, 98 # 100, 100
conf_train, conf_test = split(ds, n_train, n_test)

species = unique(vcat([atomic_symbol.(get_system(c).particles)
          for c in conf_train]...))


# Define IAP model #############################################################

# Define ACE parameters
ace = ACE(species           = [:Hf], #[:Hf, :O], 
          body_order        = 3,
          polynomial_degree = 3,
          wL                = 1.0,
          csp               = 1.0,
          r0                = 1.0,
          rcutoff           = 5.0)
@save_var path ace

# Compute energy descriptors and update training dataset
println("Computing energy descriptors of training dataset...")
e_descr_train = compute_local_descriptors(conf_train,
                                          ace,
                                          T = Float32)
ds_train = DataSet(conf_train .+ e_descr_train)

# Compute energy descriptors and update test dataset
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test,
                                         ace,
                                         T = Float32)
ds_test = DataSet(conf_test .+ e_descr_test)

# Compute training and test graphs
train_graphs = compute_graphs(ds_train,
                              rcutoff = 5.0u"Å",
                              normalize = false)
train_graphs_gpu = [g |> device for g in train_graphs]

test_graphs = compute_graphs(ds_test,
                             rcutoff = 5.0u"Å",
                             normalize = false)
test_graphs_gpu = [g |> device for g in test_graphs]

# Define GNN model for each species
n_desc = length(e_descr_train[1][1])
gnn = GNNChain(GCNConv(n_desc => 32, σ),
               GCNConv(32 => 32, σ),
               GlobalPool(mean),
               Dense(32, 1, init = Flux.glorot_uniform,
                            bias = false)) |> device


# Loss #########################################################################
energy_loss(gnn, g::GNNGraph) = Flux.mse(first(gnn(g, g.x)), g.z)
#energy_loss(gnn, g::GNNGraph) = mean((vec(gnn(g, g.x)) - g.z).^2)

# Learn ########################################################################
println("Learning energies...")

log_step = 10

opt = Flux.setup(Adam(1f-3), gnn)
n_epochs = 3000
for epoch in 1:n_epochs
    for g in train_graphs_gpu
        grad = gradient(gnn -> energy_loss(gnn, g), gnn)
        Flux.update!(opt, gnn, grad[1])
    end
    if epoch % log_step == 0
        @info (; epoch, train_loss=mean(energy_loss.([gnn], train_graphs_gpu)),
                        test_loss=mean(energy_loss.([gnn], test_graphs_gpu)))
    end
end

opt = Flux.setup(Adam(1f-5), gnn)
n_epochs = 2000
#for epoch in 1:n_epochs
train_loss = mean(energy_loss.([gnn], train_graphs_gpu))
epoch = 1
while train_loss > 0.1
    for g in train_graphs_gpu
        grad = gradient(gnn -> energy_loss(gnn, g), gnn)
        Flux.update!(opt, gnn, grad[1])
    end
    if epoch % log_step == 0
        @info (; epoch, train_loss=mean(energy_loss.([gnn], train_graphs_gpu)),
                        test_loss=mean(energy_loss.([gnn], test_graphs_gpu)))
    end
    epoch += 1
    train_loss = mean(energy_loss.([gnn], train_graphs_gpu))
end


# Post-process output: calculate metrics, create plots, and save results #######

# Get true and predicted values
n_atoms_train = length.(get_system.(ds_train))
n_atoms_test = length.(get_system.(ds_test))

e_train, e_train_pred = get_all_energies(ds_train) ./ n_atoms_train,
                        get_all_energies(train_graphs_gpu, gnn) ./ n_atoms_train
@save_var path e_train
@save_var path e_train_pred

e_test, e_test_pred = get_all_energies(ds_test) ./ n_atoms_test,
                      get_all_energies(test_graphs_gpu, gnn) ./ n_atoms_test
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

# Save training graph dataset
save("$path/train_graphs.jld", "train_graphs", train_graphs)
save("$path/test_graphs.jld", "test_graphs", test_graphs)

