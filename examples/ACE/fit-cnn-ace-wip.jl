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
include("utils/utils.jl")


# Load input parameters
args = ["experiment_path",      "a-Hfo2-300K-NVT-6000-CNN-ACE/",
        "dataset_path",         "data/",
        "dataset_filename",     "a-Hfo2-300K-NVT-6000.extxyz",
        "energy_units",         "eV",
        "distance_units",       "Å",
        "random_seed",          "100",
        "n_train_sys",          "100",
        "n_test_sys",           "100",
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
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train)
n_desc = length(e_descr_train[1][1])

# Pre-process data
xs = []
for c in ds_train
    ld_c = reduce(hcat, get_values(get_local_descriptors(c)))'
    ld_c = cat( ld_c[:, 1:n_desc÷2], ld_c[:, n_desc÷2+1:end], dims=3 )
    if xs == []
        xs = ld_c
    else
        xs = cat( xs, ld_c, dims=4)
    end
end
ys = get_all_energies(ds_train)

# Define neural network model
(n_atoms, n_basis, n_types, batch_size) = size(xs)
nn = Flux.@autosize (n_atoms, n_basis, n_types, batch_size) Chain(
    Conv((3, 3), 2=>6, relu),
    MaxPool((2, 2)),
    Conv((3, 3), _=>16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(_ => 120, relu),
    Dense(_ => 84, relu), 
    Dense(_ => 1),
)
#nace = NNIAP(nn, ace)

# Learn
println("Learning energies and forces...")
#w_e, w_f = input["w_e"], input["w_f"]
#opt = eval(Meta.parse(input["optimiser"]))
#n_epochs = input["n_epochs"]
#learn!(nace, ds_train, opt, n_epochs, loss, w_e, w_f)

η = 3e-4         # learning rate
λ = 1e-2         # for weight decay
epochs = 100_000
opt_rule = OptimiserChain(WeightDecay(λ), Adam(η))
opt_state = Flux.setup(opt_rule, nn)

for epoch in 1:epochs
    grads = Flux.gradient(m -> Flux.mse(m(xs)', ys), nn)
    Flux.update!(opt_state, nn, grads[1])
    if epoch % 100 == 0
        train_loss = Flux.mse(nn(xs)', ys)
        println("epoch = $epoch; loss = $train_loss")
    end
end

end # end of "learn_time = @elapsed begin"

@savevar path Flux.params(nace.nn)

