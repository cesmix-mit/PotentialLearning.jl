# Work in progress

using AtomsBase
using Unitful, UnitfulAtomic
using InteratomicPotentials
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra
using Flux
using Optimization
using OptimizationOptimJL
using GraphNeuralNetworks
using Random
include("utils/utils.jl")

# Load input parameters
args = ["experiment_path",      "a-Hfo2-300K-NVT-6000-GNNACE/",
        "dataset_path",         "data/",
        "dataset_filename",     "a-Hfo2-300K-NVT-6000.extxyz",
        "energy_units",         "eV",
        "distance_units",       "Å",
        "random_seed",          "100",
        "n_train_sys",          "200",
        "n_test_sys",           "200",
#        "n_red_desc",           "0", # No. of reduced descriptors. O: don't apply reduction
#        "nn",                   "Chain(Dense(n_desc,8,relu),Dense(8,1))",
#        "n_epochs",             "10000",
#        "n_batches",            "1",
#        "optimiser",            "Adam(0.01)", # e.g. Adam(0.01) or BFGS()
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
B_time = @elapsed e_descr_train = compute_local_descriptors(conf_train, ace)
println("Computing force descriptors of training dataset...")
dB_time = @elapsed f_descr_train = compute_force_descriptors(conf_train, ace)
GC.gc()
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train)
n_desc = length(e_descr_train[1][1])


# Define input graphs

function distance_3d(x1, y1, z1, x2, y2, z2)
    return sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
end

function adj_matrix(positions, threshold = 0.5u"Å")
    adjacency = Matrix{Bool}(undef, length(positions), length(positions))
    for i in 1:length(positions)
        @simd for j in 1:i
            adjacency[i, j] = distance_3d(positions[i]..., positions[j]...) < threshold
        end
    end
    return Symmetric(adjacency)
end

atom_type_id = Dict( :Hf => 1, :O => 2 )
all_graphs = []
for c in ds_train
    adj = adj_matrix(get_positions(c))
    at_ids = [ atom_type_id[a]  for a in atomic_symbol(get_system(c)) ]
    ld_mat = hcat(get_values(get_local_descriptors(c))...)
    g = GNNGraph(adj, ndata = (; x = ld_mat, y = at_ids),
                      gdata = (; z = get_values(get_energy(c))))
    push!(all_graphs, g)
end

# Define GNN model

device = Flux.cpu
model = GNNChain(GCNConv(n_desc => 64),
                 BatchNorm(64),     # Apply batch normalization on node features (nodes dimension is batch dimension)
                 x -> relu.(x),
                 GCNConv(64 => 64, relu),
                 GlobalPool(mean),  # aggregate node-wise features into graph-wise features
                 Dense(64, 1)) |> device

ps = Flux.params(model)
opt = Adam(1f-4)


# Training

# loss(g::GNNGraph) = mean((sum(model(g, g.x, g.y)) - g.z).^2)
loss(g::GNNGraph) = mean((sum(model(g, g.x)) - g.z).^2)

for epoch in 1:100
    for g in all_graphs
        g = g |> device
        grad = gradient(() -> loss(g), ps)
        Flux.Optimise.update!(opt, ps, grad)
    end

    @info (; epoch, train_loss=mean(loss.(all_graphs)))
end


