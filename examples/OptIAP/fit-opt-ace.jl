# Run this script:
#    $ cd examples/HyperLearn
#    $ julia --project=../ --threads=4
#    julia> include("fit-opt-ace.jl")


push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials, InteratomicBasisPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra
using Random
include("../utils/utils.jl")
include("HyperLearn.jl")

# Experiment folder
path = "a-HfO2-Opt/"
run(`mkdir -p $path`)

# Configuration dataset
ds_path = string("../data/a-HfO2/a-Hfo2-300K-NVT-6000.extxyz")
ds = load_data(ds_path, uparse("eV"), uparse("Å"))

# Training and test configuration datasets
n_train, n_test = 500, 500
conf_train, conf_test = split(ds, n_train, n_test)

# Dataset selector
ε, min_pts, sample_size = 0.05, 5, 10
dataset_selector = DBSCANSelector(  conf_train,
                                    ε,
                                    min_pts,
                                    sample_size)
#ss = kDPP(conf_train, GlobalMean(), DotProduct(); batch_size = 200)

# Dataset generator
dataset_generator = Nothing

# IAP model
model = ACE

# IAP parameters
model_pars = OrderedDict(
                    :species           => [[:Hf, :O]],
                    :body_order        => [2, 3, 4],
                    :polynomial_degree => [3, 4, 5],
                    :wL                => [1.0],
                    :csp               => [1.0],
                    :r0                => [1.0],
                    :rcutoff           => [4.5, 5.0, 5.5])

# Hyper-optimizer
n_samples = 18
#sampler = RandomSampler()
#sampler = LHSampler() # Requires all candidate vectors to have the same length as the number of iterations
#sampler = Hyperband(R=50, η=3, inner=RandomSampler())
sampler = Hyperband(R=18, η=3, inner=BOHB(dims=[ Hyperopt.Categorical(1),
                                                 Hyperopt.Continuous(),
                                                 Hyperopt.Continuous(),
                                                 Hyperopt.Continuous(),
                                                 Hyperopt.Continuous(),
                                                 Hyperopt.Continuous(),
                                                 Hyperopt.Continuous()]))
ho_pars = OrderedDict(:i => n_samples,
                      :sampler => sampler)

# Maximum no. of iterations
max_iterations = 1

# End condition
end_condition() = return false

# Accuracy threshold
acc_threshold = 0.1

# Weights and intercept
weights = [1.0, 1.0]
intercept = false

# Hyper-learn
hyper_optimizer =
hyperlearn!(model,
            model_pars,
            ho_pars,
            conf_train,
            dataset_selector,
            dataset_generator,
            max_iterations,
            end_condition,
            acc_threshold,
            weights,
            intercept)

# Optimal IAP
opt_iap = hyper_optimizer.minimum.opt_iap
@savevar path opt_iap.β

# Post-process output: calculate metrics, create plots, and save results

# Show optimization results
loss_pars = print(hyper_optimizer)
@savevar path loss_pars

# Plot loss vs time
loss_time = plot_loss_time(hyper_optimizer)
@savefig path loss_time

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, opt_iap.basis)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test, opt_iap.basis)
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

# Get true and predicted values
e_test, e_test_pred = get_all_energies(ds_test),
                      get_all_energies(ds_test, opt_iap)
f_test, f_test_pred = get_all_forces(ds_test),
                      get_all_forces(ds_test, opt_iap)
@savevar path e_test
@savevar path e_test_pred
@savevar path f_test
@savevar path f_test_pred

# Compute metrics
e_metrics = get_metrics(e_test_pred, e_test,
                        metrics = [mae, rmse, rsq],
                        label = "e_test")
f_metrics = get_metrics(f_test_pred, f_test,
                        metrics = [mae, rmse, rsq, mean_cos],
                        label = "f_test")
metrics = merge(e_metrics, f_metrics)
@savecsv path metrics

# Plot and save results
e_test_plot = plot_energy(e_test_pred, e_test)
f_test_plot = plot_forces(f_test_pred, f_test)
f_test_cos  = plot_cos(f_test_pred, f_test)
@savefig path e_test_plot
@savefig path f_test_plot
@savefig path f_test_cos

