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
args = ["experiment_path",      "AL-HfO2sg137n6_921/",#"a-Hfo2-300K-NVT-6000/",
        "dataset_path",         "../../../data/",
        "dataset_filename",     "HfO2sg137n6_921.xyz",#"a-Hfo2-300K-NVT-6000.extxyz",
        "random_seed",          "100",  # Random seed to ensure reproducibility of loading and subsampling.
        "n_train_sys",          "736",  #"4800",  # Training dataset size
        "n_test_sys",           "185",  #"1200",  # Test dataset size
        "n_clusters",           "20",
        "sample_size",          "3",
        "e_mae_threshold",      "0.2",
        "f_mae_threshold",      "0.2",
        "body_order",           "3",
        "polynomial_degree",    "3",
        "r0",                   "1.0",
        "rcutoff",              "7.0",
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

# Split dataset
n_train, n_test = input["n_train_sys"], input["n_test_sys"]
ds_train, ds_test = split(ds, n_train, n_test)

# Start measuring learning time
learn_time = @elapsed begin

# Calculate clusters
function PotentialLearning.get_values(pos::Vector{<:SVector})
    return [ SVector([p[i].val for i in 1:3 ]...) for p in pos]
end
function get_clusters(n_clusters, ds)
    pos = get_positions.(ds)
    X = reduce(hcat, [reduce(vcat, get_values(p)) for p in pos])
    R = kmeans(X, n_clusters)
    a = assignments(R) # get the assignments of points to clusters
    clusters = [findall(x->x==i, a) for i in 1:n_clusters]
end
n_clusters = input["n_clusters"]
clusters = get_clusters(n_clusters, ds_train)

# Define ACE parameters
species = unique(atomic_symbol(get_system(ds[1])))
body_order = input["body_order"]
polynomial_degree = input["polynomial_degree"]
wL = input["wL"]
csp = input["csp"]
r0 = input["r0"]
rcutoff = input["rcutoff"]
ace_basis = ACE(species, body_order, polynomial_degree, wL, csp, r0, rcutoff)
@savevar path ace_basis

# Active learning of ACE parameters:
# Iteratively increase training dataset and fit ACE until reach threasholds
println("Starting active learing of ACE parameters...\n")
lp = []; ds_train_cur = []; conf_train_cur = []
e_des_train_cur = []; f_des_train_cur = []
sample_size = input["sample_size"]
e_mae_threshold = input["e_mae_threshold"]
f_mae_threshold = input["f_mae_threshold"]
e_train_mae = f_train_mae = 100.0
i = 1
while e_train_mae > e_mae_threshold || f_train_mae > f_mae_threshold

    global conf_train_cur, e_des_train_cur, f_des_train_cur, ds_train_cur,
           ds_train_cur, lp, e_train_mae, f_train_mae, i

    println("Active learning iteration: $i")

    # Select new configurations by sampling from clusters
    sample(c, n) = [c[rand(1:length(c))] for _ in 1:n]
    inds = reduce(vcat, [sample(c, sample_size) for c in clusters])
    conf_new = ds[inds]
    println("New sampled configurations: $inds")
    
    # Compute energy and force descriptors of new sampled configurations
    println("Computing energy descriptors of training dataset...")
    e_des_new = compute_local_descriptors(conf_new, ace_basis)
    println("Computing force descriptors of training dataset...")
    f_des_new = compute_force_descriptors(conf_new, ace_basis)
    GC.gc()

    # Update current configurations, energy and force descriptors, and dataset
    push!(conf_train_cur, conf_new...)
    push!(e_des_train_cur, e_des_new...)
    push!(f_des_train_cur, f_des_new...)
    ds_train_cur = DataSet(conf_train_cur .+ e_des_train_cur .+ f_des_train_cur)

    # Learn ACE parameters using increased training dataset
    println("Learning energies and forces...")
    lp = LinearProblem(ds_train_cur)
    w_e = input["w_e"]; w_f = input["w_f"]
    learn!(lp, w_e, w_f) #learn!(lp)

    # Get true and predicted values
    e_train, f_train = get_all_energies(ds_train_cur), get_all_forces(ds_train_cur)
    e_train_pred, f_train_pred = get_all_energies(ds_train_cur, lp), get_all_forces(ds_train_cur, lp)
    
    # Compute metrics
    e_train_mae, e_train_rmse, e_train_rsq = calc_metrics(e_train_pred, e_train)
    f_train_mae, f_train_rmse, f_train_rsq = calc_metrics(f_train_pred, f_train)
    println("e_train_mae: $e_train_mae, e_train_rmse: $e_train_rmse, e_train_rsq: $e_train_rsq")
    println("f_train_mae: $f_train_mae, f_train_rmse: $f_train_rmse, f_train_rsq: $f_train_rsq \n")
    
    # Update iteration number
    i += 1
end

@savevar path lp.β

println("Active learning process completed.\n")

end # end of "learn_time = @elapsed begin"

# Post-process output: calculate metrics, create plots, and save results

println("Start of post-processing: generate metrics and plots")

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(ds_test, ace_basis)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(ds_test, ace_basis)
ds_test = DataSet(ds_test .+ e_descr_test .+ f_descr_test)


# Get true and predicted values
e_train, f_train = get_all_energies(ds_train_cur), get_all_forces(ds_train_cur)
e_test, f_test = get_all_energies(ds_test), get_all_forces(ds_test)
e_train_pred, f_train_pred = get_all_energies(ds_train_cur, lp), get_all_forces(ds_train_cur, lp)
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
B_time = dB_time = -1
metrics = get_metrics( e_train_pred, e_train, f_train_pred, f_train,
                       e_test_pred, e_test, f_test_pred, f_test,
                       B_time, dB_time, learn_time)
@savecsv path metrics

# Plot and save results
e_test_plot = plot_energy(e_test_pred, e_test)
@savefig path e_test_plot
f_test_plot = plot_forces_2(f_test_pred, f_test)
@savefig path f_test_plot
f_test_cos = plot_cos(f_test_pred, f_test)
@savefig path f_test_cos

