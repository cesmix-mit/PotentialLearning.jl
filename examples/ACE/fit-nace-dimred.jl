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
        "random_seed",          "100",  # Random seed to ensure reproducibility of loading and subsampling.
        "n_train_sys",          "200",  # Training dataset size
        "n_test_sys",           "200", # Test dataset size
        "nn",                   "Chain(Dense(n_desc,8,Flux.relu),Dense(8,1))",
        "epochs",               "100000",
        "n_batches",            "1",
        "optimiser",            "BFGS",
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
ds = load_data(ds_path, ExtXYZ(u"eV", u"Å"))

ds = ds[1:2000]

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
ace_basis = ACE(species, body_order, polynomial_degree, wL, csp, r0, rcutoff)
@savevar path ace_basis

# Update training dataset by adding energy and force descriptors
println("Computing energy descriptors of training dataset: ")
B_time = @elapsed begin
    e_descr_train = [LocalDescriptors(compute_local_descriptors(sys, ace_basis)) 
                                      for sys in ProgressBar(get_system.(ds_train))]
end

println("Computing force descriptors of training dataset: ")
dB_time = @elapsed begin
    f_descr_train = [ForceDescriptors([[fi[i, :] for i = 1:3]
                                       for fi in compute_force_descriptors(sys, ace_basis)])
                     for sys in ProgressBar(get_system.(ds_train))]
end

ds_train_1 = DataSet(ds_train .+ e_descr_train .+ f_descr_train)

# Dimension reduction of energy and force descriptors

# λ_pca, W_pca = fit(ds_train_1, PCA()) # Current implementation


function fit_pca(d, tol)
    m = [mean(d[:,i]) for i in 1:size(d)[2]]
    dc = reduce(hcat,[d[:,i] .- m[i] for i in 1:size(d)[2]])
    Q = Symmetric(mean(dc[i,:]*dc[i,:]' for i in 1:size(dc,1)))
    λ, ϕ = eigen(Q)
    λ, ϕ = λ[end:-1:1], ϕ[:, end:-1:1] # reorder by column
    Σ = 1.0 .- cumsum(λ) / sum(λ)
    W = ϕ[1:tol, :] # W = ϕ[:, Σ .> tol]
    return λ, W, m
end

tol = 10

lll = get_values.(get_local_descriptors.(ds_train_1))
lll_mat = Matrix(hcat(vcat(lll...)...)')
λ_l, W_l, m_l = fit_pca(lll_mat, tol)
e_descr_train_red = [LocalDescriptors([((l .- m_l)' * W_l')' for l in ll ]) for ll in lll]

fff = get_values.(get_force_descriptors.(ds_train_1))
fff_mat = Matrix(hcat(vcat(vcat(fff...)...)...)')
λ_f, W_f, m_f = fit_pca(fff_mat, tol)
f_descr_train_red = [ForceDescriptors([[((fc .- m_f)' * W_f')' for fc in f] for f in ff]) for ff in fff]

ds_train = DataSet(ds_train .+ e_descr_train_red .+ f_descr_train_red)


## Dont erase
#using MultivariateStats, Statistics
#s = open("a-Hfo2-300K-NVT-6000-NACE/locdesc.dat") do file
#    read(file, String)
#end
#a = eval(Meta.parse(s))
#b = reduce(hcat,[a[:,i] .- mean(a[:,i]) for i in 1:size(a)[2]])
#M = MultivariateStats.fit(MultivariateStats.PCA, c)
#R = predict(M, c)


#using MultivariateStats, Statistics
#a = vcat(get_values.(get_local_descriptors.(ds_train_1))...)
#b = Matrix(hcat(vcat(get_values.(get_local_descriptors.(ds_train_1))...)...)')
##c = reduce(hcat,[b[:,1] .- mean(b[:,1]) for i in 1:size(b)[2]])

#Q = Symmetric(mean(di*di' for di in foreachrow(a)))

#using LinearAlgebra

#Qa = Symmetric(Symmetric(mean(a[i,:]*a[i,:]' for i in 1:size(a,1))))
#Qc = Symmetric(Symmetric(mean(c[i,:]*c[i,:]' for i in 1:size(c,1))))

#λ, ϕ = eigen(Qa)
#λ, ϕ = λ[end:-1:1], ϕ[end:-1:1, :] # reorder

#Σ = 1.0 .- cumsum(λ) / sum(λ)
#tol = 0.00001
#W = ϕ[Σ .> tol, :]
#λ, W

################################################################################

# Define neural network model
n_desc = length(e_descr_train_red[1][1])
nn = eval(Meta.parse(input["nn"])) # e.g. Chain(Dense(n_desc,2,Flux.relu), Dense(2,1))

# Create the neural network interatomic potential
nnbp = NNBasisPotential(nn, ace_basis)

# Auxiliary functions. TODO: add this to InteratomicBasisPotentials? ###########
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
    dnndb = first(gradient(x->sum(p.nn(x)), B)) # grad_mlp(p.nn, B)
    dbdr = get_values(get_force_descriptors(c))
    return [[-dnndb ⋅ dbdr[atom][coor] for coor in 1:3]
             for atom in 1:length(dbdr)]
end

################################################################################

# Loss
function loss(nn, basis, ds, w_e = 1, w_f = 1)
    nnbp = NNBasisPotential(nn, basis)
    es, es_pred = get_all_energies(ds), get_all_energies(ds, nnbp)
    fs, fs_pred = get_all_forces(ds), get_all_forces(ds, nnbp)
    return w_e * Flux.mse(es_pred, es) + w_f * Flux.mse(fs_pred, fs)
end

# Learning functions
println("Learning energies and forces...")

# Flux.jl training
function learn!(nnbp, ds, opt::Flux.Optimise.AbstractOptimiser, epochs, loss, w_e, w_f)
    optim = Flux.setup(opt, nnbp.nn)  # will store optimiser momentum, etc.
    ∇loss(nn, basis, ds, w_e, w_f) = gradient((nn) -> loss(nn, basis, ds, w_e, w_f), nn)
    losses = []
    for epoch in 1:epochs
        # Compute gradient with current parameters and update model
        grads = ∇loss(nnbp.nn, nnbp.basis, ds, w_e, w_f)
        Flux.update!(optim, nnbp.nn, grads[1])
        # Logging
        curr_loss = loss(nnbp.nn, nnbp.basis, ds, w_e, w_f)
        push!(losses, curr_loss)
        println(curr_loss)
    end
end

# Optimization.jl training
function learn!(nnbp, ds, opt::Optim.FirstOrderOptimizer, maxiters, loss, w_e, w_f)
    ps, re = Flux.destructure(nnbp.nn)
    batchloss(ps, p) = loss(re(ps), nnbp.basis, ds_train, w_e, w_f)
    opt = BFGS()
    ∇bacthloss = OptimizationFunction(batchloss, Optimization.AutoForwardDiff()) # Optimization.AutoZygote()
    prob = OptimizationProblem(∇bacthloss, ps, []) # prob = remake(prob,u0=sol.minimizer)
    cb = function (p, l) println("Loss: $l"); return false end
    sol = solve(prob, opt, maxiters=maxiters, callback = cb) # reltol = 1e-14
    ps = sol.u
    nn = re(ps)
    global nnbp = NNBasisPotential(nn, ace_basis) # TODO: improve this
end

# Learn
w_e, w_f = input["w_e"], input["w_f"]
opt = @eval $(Symbol(input["optimiser"]))() # Flux.Adam(0.01)
epochs = input["epochs"]
learn_time = @elapsed begin
    learn!(nnbp, ds, opt, epochs, loss, w_e, w_f)
end

@savevar path Flux.params(nnbp.nn)

## Post-process output: calculate metrics, create plots, and save results
println("Post-processing...")

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset: ")
e_descr_test = [LocalDescriptors(compute_local_descriptors(sys, ace_basis)) 
                for sys in ProgressBar(get_system.(ds_test))]

println("Computing force descriptors of test dataset: ")
f_descr_test = [ForceDescriptors([[fi[i, :] for i = 1:3]
                                   for fi in compute_force_descriptors(sys, ace_basis)])
                for sys in ProgressBar(get_system.(ds_test))]

ds_test_1 = DataSet(ds_test .+ e_descr_test .+ f_descr_test)

# Dimension reduction of energy and force descriptors
#λ_pca, W_pca = fit(ds_test_1, PCA())
#lll = get_values.(get_local_descriptors.(ds_test_1))
#e_descr_test_red = [LocalDescriptors([W_pca * l for l in ll ]) for ll in lll]
#fff = get_values.(get_force_descriptors.(ds_test_1))
#f_descr_test_red = [ForceDescriptors([[W_pca * fc for fc in f] for f in ff]) for ff in fff]

lll = get_values.(get_local_descriptors.(ds_test_1))
lll_mat = Matrix(hcat(vcat(lll...)...)')
λ_l, W_l, m_l = fit_pca(lll_mat, tol)
e_descr_test_red = [LocalDescriptors([((l .- m_l)' * W_l')' for l in ll ]) for ll in lll]

fff = get_values.(get_force_descriptors.(ds_test_1))
fff_mat = Matrix(hcat(vcat(vcat(fff...)...)...)')
λ_f, W_f, m_f = fit_pca(fff_mat, tol)
f_descr_test_red = [ForceDescriptors([[((fc .- m_f)' * W_f')' for fc in f] for f in ff]) for ff in fff]


ds_test = DataSet(ds_test .+ e_descr_test_red .+ f_descr_test_red)

# Get true and predicted values
e_train, f_train = get_all_energies(ds_train), get_all_forces(ds_train)
e_test, f_test = get_all_energies(ds_test), get_all_forces(ds_test)
e_train_pred, f_train_pred = get_all_energies(ds_train, nnbp), get_all_forces(ds_train, nnbp)
e_test_pred, f_test_pred = get_all_energies(ds_test, nnbp), get_all_forces(ds_test, nnbp)

@savevar path e_train
@savevar path e_train_pred
@savevar path f_train
@savevar path f_train_pred
@savevar path e_test
@savevar path e_test_pred
@savevar path f_test
@savevar path f_test_pred

metrics = get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                       e_test_pred, e_test, f_test_pred, f_test,
                       B_time, dB_time, learn_time)
@savecsv path metrics

e_test_plot = plot_energy(e_test_pred, e_test)
@savefig path e_test_plot

f_test_plot = plot_forces(f_test_pred, f_test)
@savefig path f_test_plot

f_test_cos = plot_cos(f_test_pred, f_test)
@savefig path f_test_cos

