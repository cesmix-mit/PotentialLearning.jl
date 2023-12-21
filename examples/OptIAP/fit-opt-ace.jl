# Run this script:
#    $ cd examples/HyperLearn
#    $ julia --project=../ --threads=4
#    julia> include("fit-opt-ace.jl")


push!(Base.LOAD_PATH, "../../")

using AtomsBase
using InteratomicPotentials
using PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra, DataFrames
using Random
include("../utils/utils.jl")
include("HyperLearn.jl")


# Setup experiment #############################################################

# Experiment folder
path = "a-HfO2-Opt/"
run(`mkdir -p $path`)


# Define training and test configuration datasets ##############################

# Load complete configuration dataset
ds_path = string("../data/a-HfO2/a-Hfo2-300K-NVT-6000.extxyz")
ds = load_data(ds_path, uparse("eV"), uparse("Å"))

# Split configuration dataset into training and test
n_train, n_test = 500, 500
conf_train, conf_test = split(ds, n_train, n_test)


# Define dataset subselector ###################################################

# Subselector, option 1: RandomSelector
#dataset_selector = RandomSelector(length(conf_train); batch_size = 100)

# Subselector, option 2: DBSCANSelector
#ε, min_pts, sample_size = 0.05, 3, 100
#dataset_selector = DBSCANSelector(conf_train,
#                                  ε,
#                                  min_pts,
#                                  sample_size)

# Subselector, option 3: kDPP + ACE (requires calculation of energy descriptors)
basis = ACE(species           = [:Hf, :O],
            body_order        = 2,
            polynomial_degree = 3,
            wL                = 1.0,
            csp               = 1.0,
            r0                = 1.0,
            rcutoff           = 5.0)
e_descr = compute_local_descriptors(conf_train,
                                    basis,
                                    pbar = false)
conf_train_kDPP = DataSet(conf_train .+ e_descr)
dataset_selector = kDPP(  conf_train_kDPP,
                          GlobalMean(),
                          DotProduct();
                          batch_size = 100)

# Subsample trainig dataset
inds = get_random_subset(dataset_selector)
conf_train = conf_train[inds]
GC.gc()

# Define parameters to compute optimal sample size (not used for now)
max_iterations = 1
end_condition() = return false


# Define IAP model and candidate parameters ####################################

# Define IAP model
model = ACE

# Define IAP parameters to be optimized
model_pars = OrderedDict(
                    :species           => [[:Hf, :O]],
                    :body_order        => [2, 3, 4],
                    :polynomial_degree => [3, 4, 5],
                    :rcutoff           => [4.5, 5.0, 5.5],
                    :wL                => [0.5, 1.0, 1.5],
                    :csp               => [0.5, 1.0, 1.5],
                    :r0                => [0.5, 1.0, 1.5])

# Define hyper-optimizer parameters ############################################

# Sampler, option 1: RandomSampler
#sampler = RandomSampler()

# Sampler, option 2: LHSampler (requires all candidate vectors to have the same length as the number of iterations)
#sampler = LHSampler()

# Sampler, option 3: Hyperband + RandomSampler()
#sampler = Hyperband(R=10, η=3, inner=RandomSampler())

# Sampler, option 4: Hyperband + BOHB
sampler = Hyperband(R=10, η=3, inner=BOHB(dims=[ Hyperopt.Categorical(1),
                                                 Hyperopt.Continuous(),
                                                 Hyperopt.Continuous(),
                                                 Hyperopt.Continuous(),
                                                 Hyperopt.Continuous(),
                                                 Hyperopt.Continuous(),
                                                 Hyperopt.Continuous()]))


n_samples = 20

ho_pars = OrderedDict(:i => n_samples,
                      :sampler => sampler)

e_mae_max = 0.05
f_mae_max = 0.05

# Define linear solver parameters
weights = [1.0, 1.0]
intercept = true

# Perform hyper-parameter optimization #########################################

hyper_optimizer =
hyperlearn!(conf_train,
            model,
            model_pars,
            ho_pars;
            e_mae_max = e_mae_max,
            f_mae_max = f_mae_max,
            weights   = weights,
            intercept = intercept)

# Post-process output: calculate metrics, create plots, and save results #######

# Optimal IAP
opt_iap = hyper_optimizer.minimum.opt_iap
@save_var path opt_iap.β
@save_var path opt_iap.β0
@save_var path opt_iap.basis

# Prnt and save optimization results
results = get_results(hyper_optimizer)
println(results)
@save_dataframe path results

# Plot parameter values vs accuracy
pars_acc_plot = plot(hyper_optimizer)
@save_fig path pars_acc_plot

# Plot loss vs time
acc_time = plot_acc_time(hyper_optimizer)
@save_fig path acc_time

# Update training dataset by adding energy and force descriptors
println("Computing energy descriptors of training dataset...")
e_descr_train = compute_local_descriptors(conf_train, opt_iap.basis)
println("Computing force descriptors of training dataset...")
f_descr_train = compute_force_descriptors(conf_train, opt_iap.basis)
ds_train = DataSet(conf_train .+ e_descr_train .+ f_descr_train)

# Update test dataset by adding energy and force descriptors
println("Computing energy descriptors of test dataset...")
e_descr_test = compute_local_descriptors(conf_test, opt_iap.basis)
println("Computing force descriptors of test dataset...")
f_descr_test = compute_force_descriptors(conf_test, opt_iap.basis)
ds_test = DataSet(conf_test .+ e_descr_test .+ f_descr_test)

GC.gc()

# Get true and predicted values
n_atoms_train = length.(get_system.(ds_train))
n_atoms_test = length.(get_system.(ds_test))

e_train, e_train_pred = get_all_energies(ds_train) ./ n_atoms_train,
                        get_all_energies(ds_train, opt_iap) ./ n_atoms_train
f_train, f_train_pred = get_all_forces(ds_train),
                        get_all_forces(ds_train, opt_iap)
@save_var path e_train
@save_var path e_train_pred
@save_var path f_train
@save_var path f_train_pred

e_test, e_test_pred = get_all_energies(ds_test) ./ n_atoms_test,
                      get_all_energies(ds_test, opt_iap) ./ n_atoms_test
f_test, f_test_pred = get_all_forces(ds_test),
                      get_all_forces(ds_test, opt_iap)
@save_var path e_test
@save_var path e_test_pred
@save_var path f_test
@save_var path f_test_pred

# Compute metrics
e_train_metrics = get_metrics(e_train, e_train_pred,
                              metrics = [mae, rmse, rsq],
                              label = "e_train")
f_train_metrics = get_metrics(f_train, f_train_pred,
                              metrics = [mae, rmse, rsq, mean_cos],
                              label = "f_train")
train_metrics = merge(e_train_metrics, f_train_metrics)
@save_dict path train_metrics

e_test_metrics = get_metrics(e_test, e_test_pred,
                             metrics = [mae, rmse, rsq],
                             label = "e_test")
f_test_metrics = get_metrics(f_test, f_test_pred,
                             metrics = [mae, rmse, rsq, mean_cos],
                             label = "f_test")
test_metrics = merge(e_test_metrics, f_test_metrics)
@save_dict path test_metrics

# Plot and save results

e_plot = plot_energy(e_train, e_train_pred,
                     e_test, e_test_pred)
@save_fig path e_plot

e_train_plot = plot_energy(e_train, e_train_pred)
f_train_plot = plot_forces(f_train, f_train_pred)
f_train_cos  = plot_cos(f_train, f_train_pred)
@save_fig path e_train_plot
@save_fig path f_train_plot
@save_fig path f_train_cos

e_test_plot = plot_energy(e_test, e_test_pred)
f_test_plot = plot_forces(f_test, f_test_pred)
f_test_cos  = plot_cos(f_test, f_test_pred)
@save_fig path e_test_plot
@save_fig path f_test_plot
@save_fig path f_test_cos

