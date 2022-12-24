using AtomsBase
using Unitful, UnitfulAtomic
using InteratomicPotentials 
using InteratomicBasisPotentials
using PotentialLearning
using LinearAlgebra
using Random
using ProgressBars

# Load input parameters
args = ["experiment_path",      "a-Hfo2-300K-NVT-6000/",
        "dataset_path",         "../../../data/",
        "dataset_filename",     "a-Hfo2-300K-NVT-6000.extxyz",
        "random_seed",          "0",   # Random seed to ensure reproducibility of loading and subsampling.
        "split_prop",           "0.8", # 80% training, 20% test.
        "max_train_sys",        "800", # Subsamples up to 800 systems from the training dataset.
        "max_test_sys",         "200", # Subsamples up to 200 systems from the test dataset.
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
ds_path = dirname(@__DIR__)*"/data/"*input["dataset_filename"]
ds = load_data(ds_path, ExtXYZ(u"eV", u"Å"))

# Subsample datasets
ds_train, ds_test = ds[1:10], ds[11:20]

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

# Computing descriptors 
print("Computing energy descriptors of train dataset: ")
B_time = @elapsed e_descr_train = [LocalDescriptors(compute_local_descriptors(sys, ace)) 
                                   for sys in ProgressBar(get_system.(ds_train))]

print("Computing force descriptors of train dataset: ")
dB_time = @elapsed f_descr_train = [ForceDescriptors([[fi[i, :] for i = 1:3]
                                                       for fi in compute_force_descriptors(sys, ace)])
                                    for sys in ProgressBar(get_system.(ds_train))]

print("Computing energy descriptors of test dataset: ")
e_descr_test = [LocalDescriptors(compute_local_descriptors(sys, ace)) 
                for sys in ProgressBar(get_system.(ds_test))]

print("Computing force descriptors of test dataset: ")
f_descr_test = [ForceDescriptors([[fi[i, :] for i = 1:3]
                                   for fi in compute_force_descriptors(sys, ace)])
                for sys in ProgressBar(get_system.(ds_test))]

# Learn
learn_time = @elapsed begin
    ds_train = DataSet( Configuration.(get_energy.(ds_train), e_descr_train,
                                       get_forces.(ds_train), f_descr_train))
    lp = LinearProblem(ds_train)
    learn!(lp)
end

ds_test = DataSet( Configuration.(get_energy.(ds_test), e_descr_test,
                                  get_forces.(ds_test), f_descr_test))


# Post-process output: calculate metrics, create plots, and save results

# The following code needs to be improved

function get_energy_vals(ds)
    return get_values.(get_energy.(ds))
end

function get_energy_pred_vals(ds)
    return [B ⋅ lp.β for B in compute_feature.(get_local_descriptors.(ds), [GlobalSum()])]
end

function get_forces_vals(ds)
    return vcat(vcat(get_values.(get_forces.(ds))...)...)
end

function get_forces_pred_vals(ds)
    force_descriptors = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds]
    return vcat([dB' * lp.β for dB in [reduce(hcat, fi) for fi in force_descriptors]]...)
end

e_train = get_energy_vals(ds_train)
e_train_pred = get_energy_pred_vals(ds_train)
f_train = get_forces_vals(ds_train)
f_train_pred = get_forces_pred_vals(ds_train)

e_test = get_energy_vals(ds_test)
e_test_pred = get_energy_pred_vals(ds_test)
f_test = get_forces_vals(ds_test)
f_test_pred = get_forces_pred_vals(ds_test)

@savevar path e_train
@savevar path e_train_pred
@savevar path f_train
@savevar path f_train_pred
@savevar path e_test
@savevar path e_test_pred
@savevar path f_test
@savevar path f_test_pred

#e_train_pred = B_train * lp.β
#f_train_pred = dB_train * lp.β
#e_test_pred = B_test * lp.β
#f_test_pred = dB_test * lp.β

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

