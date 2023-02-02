using AtomsBase
using Unitful, UnitfulAtomic
using InteratomicPotentials 
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra
using Random
using ProgressBars
using Clustering
using StaticArrays
include("utils.jl")

# Load input parameters
args = ["experiment_path",      "a-Hfo2-300K-NVT-6000-37/",
        "dataset_path",         "../../../data/",
        "dataset_filename",     "a-Hfo2-300K-NVT-6000.extxyz",
        "random_seed",          "100",  # Random seed to ensure reproducibility of loading and subsampling.
        "n_train_sys",          "100",  # Training dataset size
        "n_test_sys",           "100",  # Test dataset size
        "n_body",               "3",
        "max_deg",              "4",
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

# Define clusters
using StaticArrays
function PotentialLearning.get_values(pos::Vector{<:SVector})
    return [ SVector([p[i].val for i in 1:3 ]...) for p in pos]
end
function sample(c, n)
    return [ c[rand(1:length(c))] for _ in 1:n]
end
pos = get_positions.(ds_train)
flat_pos = [ reduce(vcat, get_values(p)) for p in pos ]
X = reduce(hcat, flat_pos)
R = kmeans(X, 10)
a = assignments(R) # get the assignments of points to clusters
c = counts(R) # get the cluster sizes
M = R.centers # get the cluster centers
clusters = [findall(x->x==i, a) for i in 1:10]

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

function learn_old!(lp)
    w_e = 1; w_f = 1
    
    B_train = reduce(hcat, lp.B)'
    dB_train = reduce(hcat, lp.dB)'
    e_train, f_train = get_all_energies(ds_train), get_all_forces(ds_train)
    
    # Calculate A and b.
    A = [B_train; dB_train]
    b = [e_train; f_train]

    # Calculate coefficients β.
    Q = Diagonal([w_e * ones(length(e_train));
                  w_f * ones(length(f_train))])
    β = (A'*Q*A) \ (A'*Q*b)

    copyto!(lp.β, β)
end

# increase training dataset and fit ACE until reach threasholds
curr_e_descr_train = []
curr_f_descr_train = []
curr_ds_train = []
while e_train_mae > 1.0 || f_train_mae > 1.0 

    # Sample from clusters
    inds = reduce(vcat, [sample(c, 5) for c in clusters])
    new_ds_train = ds_train[inds]
    
    # Update training dataset by adding energy and force descriptors
    println("Computing energy descriptors of training dataset: ")
    B_time = @elapsed begin
        new_e_descr_train = [LocalDescriptors(compute_local_descriptors(sys, ace_basis)) 
                                          for sys in ProgressBar(get_system.(new_ds_train))]
    end

    println("Computing force descriptors of training dataset: ")
    dB_time = @elapsed begin
        new_f_descr_train = [ForceDescriptors([[fi[i, :] for i = 1:3]
                                           for fi in compute_force_descriptors(sys, ace_basis)])
                         for sys in ProgressBar(get_system.(new_ds_train))]
    end

    push!(curr_e_descr_train, new_e_descr_train...)
    push!(curr_f_descr_train, new_f_descr_train...)
    if length(curr_ds_train) > 0
        curr_ds_train = curr_ds_train .+ new_ds_train
    else
        curr_ds_train = new_ds_train
    end
    ds_train = DataSet(curr_ds_train .+ curr_e_descr_train .+ curr_f_descr_train)

    # Learn
    println("Learning energies and forces...")
    learn_time = @elapsed begin
        lp = LinearProblem(ds_train)
        learn_old!(lp)
    end

    # Get true and predicted values
    e_train, f_train = get_all_energies(ds_train), get_all_forces(ds_train)
    e_train_pred, f_train_pred = get_all_energies(ds_train, lp), get_all_forces(ds_train, lp)
    
    # Compute metrics
    e_train_mae, e_train_rmse, e_train_rsq = calc_metrics(e_train_pred, e_train)
    f_train_mae, f_train_rmse, f_train_rsq = calc_metrics(f_train_pred, f_train)
       
end

# Post-process output: calculate metrics, create plots, and save results

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset: ")
e_descr_test = [LocalDescriptors(compute_local_descriptors(sys, ace_basis)) 
                for sys in ProgressBar(get_system.(ds_test))]

println("Computing force descriptors of test dataset: ")
f_descr_test = [ForceDescriptors([[fi[i, :] for i = 1:3]
                                   for fi in compute_force_descriptors(sys, ace_basis)])
                for sys in ProgressBar(get_system.(ds_test))]

ds_test = DataSet(ds_test .+ e_descr_test .+ f_descr_test)


# Get true and predicted values
e_train, f_train = get_all_energies(ds_train), get_all_forces(ds_train)
e_test, f_test = get_all_energies(ds_test), get_all_forces(ds_test)
e_train_pred, f_train_pred = get_all_energies(ds_train, lp), get_all_forces(ds_train, lp)
e_test_pred, f_test_pred = get_all_energies(ds_test, lp), get_all_forces(ds_test, lp)
@savevar path e_train
@savevar path e_train_pred
@savevar path f_train
@savevar path f_train_pred
@savevar path e_test
@savevar path e_test_pred
@savevar path f_test
@savevar path f_test_pred

# Compute metrics
metrics = get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                       e_test_pred, e_test, f_test_pred, f_test,
                       B_time, dB_time, learn_time)
@savecsv path metrics

# Plot and save results
e_test_plot = plot_energy(e_test_pred, e_test)
@savefig path e_test_plot
f_test_plot = plot_forces(f_test_pred, f_test)
@savefig path f_test_plot
f_test_cos = plot_cos(f_test_pred, f_test)
@savefig path f_test_cos

