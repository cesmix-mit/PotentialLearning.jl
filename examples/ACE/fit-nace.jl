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
using ProgressBars
include("utils.jl")

# Load input parameters
args = ["experiment_path",      "a-Hfo2-300K-NVT-6000-NACE/",
        "dataset_path",         "../../../data/",
        "dataset_filename",     "a-Hfo2-300K-NVT-6000.extxyz",
        "random_seed",          "0",   # Random seed to ensure reproducibility of loading and subsampling.
        "n_train_sys",          "200", # Training dataset size
        "n_test_sys",           "200", # Test dataset size
        "nn",                   "Chain(Dense(n_desc,3,Flux.sigmoid),Dense(3,1))",
        "n_epochs",             "1",
        "n_batches",            "1",
        "optimiser",            "BFGS",
        "max_it",               "1000",
        "n_body",               "2",
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
ds = load_data(ds_path, ExtXYZ(u"eV", u"Å"))

ds = ds[1:400]

# Split dataset
n_train, n_test = input["n_train_sys"], input["n_test_sys"]
ds_train, ds_test = split(ds, n_train, n_test)

# Define ACE parameters
species = unique(atomic_symbol(get_system(ds[1])))
body_order = input["n_body"]
polynomial_degree = input["max_deg"]
wL = input["wL"]
csp = input["csp"]
r0 = input["r0"]
rcutoff = input["rcutoff"]
ace = ACE(species, body_order, polynomial_degree, wL, csp, r0, rcutoff)
@savevar path ace

# Update training dataset by adding energy and force descriptors
println("Computing energy descriptors of training dataset: ")
B_time = @elapsed begin
    e_descr_train = [LocalDescriptors(compute_local_descriptors(sys, ace)) 
                                      for sys in ProgressBar(get_system.(ds_train))]
end

println("Computing force descriptors of training dataset: ")
dB_time = @elapsed begin
    f_descr_train = [ForceDescriptors([[fi[i, :] for i = 1:3]
                                       for fi in compute_force_descriptors(sys, ace)])
                     for sys in ProgressBar(get_system.(ds_train))]
end

ds_train = DataSet(ds_train .+ e_descr_train .+ f_descr_train)

# Define neural network model
n_desc = length(e_descr_train[1][1])
nn = eval(Meta.parse(input["nn"])) # e.g. Chain(Dense(n_desc,2,Flux.relu), Dense(2,1))
ps, re = Flux.destructure(nn)


# Auxiliary functions
function InteratomicPotentials.potential_energy(c::Configuration, p::NNBasisPotential)
    B = sum(get_values(get_local_descriptors(c)))
    return sum(p.nn(B))
end

#function grad_mlp(m, x0)
#    ps = Flux.params(m)
#    dsdy(x) = x>0 ? 1 : 0 # Flux.σ(x) * (1 - Flux.σ(x)) 
#    prod = 1; x = x0
#    n_layers = length(ps) ÷ 2
#    for i in 1:2:2(n_layers-1)-1  # i : 1, 3
#        y = ps[i] * x + ps[i+1]
#        x = Flux.relu.(y) # Flux.σ.(y)
#        prod = dsdy.(y) .* ps[i] * prod
#    end
#    i = 2(n_layers)-1 
#    prod = ps[i] * prod
#    return prod #reshape(prod, :)
#end

function InteratomicPotentials.force(c::Configuration, p::NNBasisPotential)
    B = sum(get_values(get_local_descriptors(c)))
    dnndb = first(gradient(x->sum(p.nn(x)), B))
    #dnndb = grad_mlp(p.nn, B)
    #dnndb = B[1:12]
    dbdr = get_values(get_force_descriptors(c))
    return [[-dnndb ⋅ dbdr[atom][coor] for coor in 1:3]
             for atom in 1:length(dbdr)]
end

# Loss
function loss(ps, ds)
    nnbp = NNBasisPotential(re(ps), ace)

    es =      [get_values(get_energy(ds[c])) for c in 1:length(ds)]
    es_pred = [potential_energy(ds[c], nnbp) for c in 1:length(ds)]

    fs =      reduce(vcat,reduce(vcat,[get_values(get_forces(ds[c])) for c in 1:length(ds)]))
    fs_pred = reduce(vcat,reduce(vcat,[force(ds[c], nnbp)            for c in 1:length(ds)]))

    #es = get_values.(get_energy.(ds))
    #es_pred = potential_energy.(ds, [nnbp])
    
    #fs = vcat(vcat(get_values.(get_forces.(ds))...)...)
    #fs_pred = vcat(vcat(force.(ds, [nnbp])...)...)
    
    w_e = 1; w_f = 1; 
    #return w_e * Flux.mse(es_pred, es)
    return w_e * Flux.mse(es_pred, es) #+ w_f * Flux.mse(fs_pred, fs)
end


losses = []
optim = Flux.setup(Flux.Adam(0.01), nn)  # will store optimiser momentum, etc.
for epoch in 1:10
        loss1, grads = Flux.withgradient(nn, ds_train, ace) do m,ds,ace
            nnbp = NNBasisPotential(m, ace)
            es =      [get_values(get_energy(ds[c])) for c in 1:length(ds)]
            es_pred = [potential_energy(ds[c], nnbp) for c in 1:length(ds)]
            
            fs =      reduce(vcat,reduce(vcat,[get_values(get_forces(ds[c])) for c in 1:length(ds)]))
            fs_pred = reduce(vcat,reduce(vcat,[force(ds[c], nnbp)            for c in 1:length(ds)]))
            
            w_e = 1; w_f = 1;
            return w_e * Flux.mse(es_pred, es) + w_f * Flux.mse(fs_pred, fs)
        end
        Flux.update!(optim, nn, grads[1])
        push!(losses, loss1)  # logging, outside gradient context
        println(loss1)
end


## Learn
#println("Training energies and forces...")

#opt = @eval $(Symbol(input["optimiser"]))()
#max_it = input["max_it"]
#w_e, w_f = input["w_e"], input["w_f"]

#lp = PotentialLearning.LearningProblem(ds_train, loss, ps)

#learn_time = @elapsed begin
#    learn!(lp; num_steps = max_it, opt = [opt])
#end

#nnbp = NNBasisPotential(re(ps), ace)

#grads = lp.∇logprob(lp.params, lp.ds)
#Flux.Optimise.update!(opt, lp.params, grads)

#model = Chain(Dense(1 => 23, tanh), Dense(23 => 1, bias=false), only)

#optim = Flux.setup(Adam(), nn)
#for epoch in 1:1000
#  Flux.train!(loss(ps, ds), nn, ds, optim)
#end



## Post-process output: calculate metrics, create plots, and save results

## Update test dataset by adding energy and force descriptors
#println("Computing energy descriptors of test dataset: ")
#e_descr_test = [LocalDescriptors(compute_local_descriptors(sys, ace)) 
#                for sys in ProgressBar(get_system.(ds_test))]

#println("Computing force descriptors of test dataset: ")
#f_descr_test = [ForceDescriptors([[fi[i, :] for i = 1:3]
#                                   for fi in compute_force_descriptors(sys, ace)])
#                for sys in ProgressBar(get_system.(ds_test))]

#ds_test = DataSet(ds_test .+ e_descr_test .+ f_descr_test)


## Get true and predicted values
#e_train, f_train = get_true_values(ds_train)
#e_test, f_test = get_true_values(ds_test)
#e_train_pred, f_train_pred = get_pred_values(lp, ds_train)
#e_test_pred, f_test_pred = get_pred_values(lp, ds_test)
#@savevar path e_train
#@savevar path e_train_pred
#@savevar path f_train
#@savevar path f_train_pred
#@savevar path e_test
#@savevar path e_test_pred
#@savevar path f_test
#@savevar path f_test_pred

#metrics = get_metrics( e_train_pred, e_train, f_train_pred, f_train,
#                       e_test_pred, e_test, f_test_pred, f_test,
#                       B_time, dB_time, learn_time)
#@savecsv path metrics

#e_test_plot = plot_energy(e_test_pred, e_test)
#@savefig path e_test_plot

#f_test_plot = plot_forces(f_test_pred, f_test)
#@savefig path f_test_plot

#f_test_cos = plot_cos(f_test_pred, f_test)
#@savefig path f_test_cos



