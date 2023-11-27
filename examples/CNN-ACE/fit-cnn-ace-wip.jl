# Run this script:
#   $ cd examples/ACE
#   $ julia --project=../ --threads=4
#   julia> include("fit-cnn-ace-wip.jl")

push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials, InteratomicBasisPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using Random
include("../utils/utils.jl")


# Load input parameters
args = ["experiment_path",      "a-HfO2-CNN-ACE-WIP/",
        "dataset_path",         "../data/a-HfO2/",
        "dataset_filename",     "a-Hfo2-300K-NVT-6000.extxyz",
        "energy_units",         "eV",
        "distance_units",       "Å",
        "random_seed",          "100",
        "n_train_sys",          "1000",
        "n_test_sys",           "1000",
        "n_red_desc",           "0", # No. of reduced descriptors. O: don't apply reduction
#        "nn",                   "Chain(Dense(n_desc,8,relu),Dense(8,1))",
#        "n_epochs",             "100",
#        "n_batches",            "1",
#        "optimiser",            "Adam(0.001)", # e.g. Adam(0.01) or BFGS()
        "n_body",               "3",
        "max_deg",              "3",
        "r0",                   "1.0",
        "rcutoff",              "5.0",
        "wL",                   "1.0",
        "csp",                  "1.0"
#        "w_e",                  "1.0",
#        "w_f",                  "1.0"
        ]
args = length(ARGS) > 0 ? ARGS : args
input = get_input(args)

# Create experiment folder
path = input["experiment_path"]
run(`mkdir -p $path`)
#@save_csv path input

# Fix random seed
if "random_seed" in keys(input)
    Random.seed!(input["random_seed"])
end

# Load dataset
ds_path = input["dataset_path"]*input["dataset_filename"] # dirname(@__DIR__)*"/data/"*input["dataset_filename"]
energy_units, distance_units = uparse(input["energy_units"]), uparse(input["distance_units"])
ds = load_data(ds_path, energy_units, distance_units)

# Split dataset
function Base.split(ds, n, m)
    ii = randperm(length(ds))
    return @views ds[first(ii, n)], ds[last(ii, m)]
end
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
B_time = @elapsed e_descr_train = compute_local_descriptors(conf_train, ace, T = Float32)
println("Computing force descriptors of training dataset...")
dB_time = @elapsed f_descr_train = compute_force_descriptors(conf_train, ace, T = Float32)
GC.gc()
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train)
n_desc = length(e_descr_train[1][1])

# Dimension reduction of energy and force descriptors of training dataset
reduce_descriptors = input["n_red_desc"] > 0
if reduce_descriptors
    n_desc = input["n_red_desc"]
    pca = PCAState(tol = n_desc)
    fit!(ds_train, pca)
    transform!(ds_train, pca)
end

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, ace, T = Float32)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test, ace, T = Float32)
GC.gc()
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

# Dimension reduction of energy and force descriptors of test dataset
if reduce_descriptors
    transform!(ds_test, pca)
end

# Aux. functions

function PotentialLearning.get_all_energies(ds::DataSet, nniap::NNIAP)
    return nniap.nns(get_e_descr_batch(ds))'
end

function get_e_descr_batch(ds)
    xs = []
    for c in ds
        ld_c = reduce(hcat, get_values(get_local_descriptors(c)))'
        #ld_c = ld_c[randperm(size(ld_c,1)),:]
        ld_c = cat( ld_c[:, 1:n_desc÷2], ld_c[:, n_desc÷2+1:end], dims=3 )
        if xs == []
            xs = ld_c
        else
            xs = cat(xs, ld_c, dims=4)
        end

#        ld_c = get_values(get_local_descriptors(c))
#        #ld_c = ld_c[randperm(length(ld_c))]
#        ld_c = cat( [Matrix(hcat(l[1:n_desc÷2], l[n_desc÷2+1:end])')
#                     for l in ld_c]..., dims=3)
#        
#        if xs == []
#            xs = ld_c
#        else
#            xs = cat(xs, ld_c, dims=4)
#        end

    end
    return xs
end

sqnorm(x) = sum(abs2, x)
function loss(x, y)
    return Flux.mse(x, y)
end

#function learn!(cnnnace, ds_train, opt, n_epochs, loss)
#    es = get_all_energies(ds_train) |> gpu
#    ld = get_e_descr_batch(ds_train) |> gpu
#    nn = cnnnace.nns |> gpu
#    opt = opt |> gpu
#    for epoch in 1:n_epochs
#        #grads = Flux.gradient(m -> loss(m(ld)', es) + sum(sqnorm, Flux.params(m)), nn)
#        grads = Flux.gradient(m -> loss(m(ld)', es), nn)
#        Flux.update!(opt, nn, grads[1])
#        if epoch % 100 == 0
#            #train_loss = loss(nn(ld)', es) + sum(sqnorm, Flux.params(nn))
#            train_loss = loss(nn(ld)', es)
#            println("epoch = $epoch; loss = $train_loss")
#        end
#    end
#    cnnnace.nns = nns |> cpu
#end

function learn!(cnnace, ds_train, ds_test, opt, n_epochs, loss)
    es = get_all_energies(ds_train) |> gpu
    ld = get_e_descr_batch(ds_train) |> gpu
    es_test = get_all_energies(ds_test) |> gpu
    ld_test = get_e_descr_batch(ds_test) |> gpu
    nns = cnnace.nns |> gpu
    opt = opt |> gpu
    for epoch in 1:n_epochs
        grads = Flux.gradient(m -> loss(m(ld)', es), nns)
        Flux.update!(opt, nns, grads[1])
        if epoch % 500 == 0
            train_loss = loss(nns(ld)', es)
            test_loss = loss(nns(ld_test)', es_test)
            println("epoch = $epoch; train loss = $train_loss, test loss = $test_loss")
        end
    end
    cnnace.nns = nns |> cpu
end

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
#w_e, w_f = input["w_e"], input["w_f"]
#opt = eval(Meta.parse(input["optimiser"]))
#n_epochs = input["n_epochs"]
#learn!(nace, ds_train, opt, n_epochs, loss, w_e, w_f)

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
n_epochs = 10_000
learn!(cnnace, ds_train, ds_test, opt_state, n_epochs, loss)
@save_var path Flux.params(cnnace.nns)

# Post-process output: calculate metrics, create plots, and save results

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


