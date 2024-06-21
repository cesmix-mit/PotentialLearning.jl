# # Optimize ACE hyper-parameters: minimize force time and fitting error.

# ## a. Load packages, define paths, and create experiment folder.

# Load packages.
using AtomsBase, InteratomicPotentials, PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra, Random, DisplayAs
using DataFrames, Hyperopt

# Define paths.
path = joinpath(dirname(pathof(PotentialLearning)), "../examples/Opt-ACE-aHfO2")
ds_path =  "$path/../data/a-HfO2/a-HfO2-300K-NVT-6000.extxyz"
res_path = "$path/results/";

# Load utility functions.
include("$path/../utils/utils.jl")

# Create experiment folder.
run(`mkdir -p $res_path`);

# ## b. Load atomistic dataset and split it into training and test.

# Load atomistic dataset: atomistic configurations (atom positions, geometry, etc.) + DFT data (energies, forces, etc.)
ds = load_data(ds_path, uparse("eV"), uparse("Å"))[1:1000] # Only the first 1K samples are used in this example.

# Split atomistic dataset into training and test
n_train, n_test = 50, 50 # Only 50 samples per dataset are used in this example.
conf_train, conf_test = split(ds, n_train, n_test)


# NEW: utilizty functions #######################################################

function estimate_time(confs, iap; batch_size = 30)
    if length(confs) < batch_size
        batch_size = length(confs)
    end
    random_selector = RandomSelector(length(confs), batch_size)
    inds = PotentialLearning.get_random_subset(random_selector)
    time = @elapsed begin
        f_descr = compute_force_descriptors(confs[inds],
                                            iap.basis,
                                            pbar = false)
        ds = DataSet(confs[inds] .+ f_descr)
        f_pred = get_all_forces(ds, iap)
    end
    n_atoms = sum(length(get_system(c)) for c in confs[inds])
    return time / n_atoms
end

function get_results(ho)
    column_names = string.(vcat(keys(ho.results[1][2])..., ho.params...))
    rows = [[values(r[2])..., p...] for (r, p) in zip(ho.results, ho.history)]
    results = DataFrame([Any[] for _ in 1:length(column_names)], column_names)
    [push!(results, r) for r in rows]
    return sort!(results)
end

function plot_err_time(ho)
    error      = [r[2][:error] for r in ho.results]
    times      = [r[2][:time_us] for r in ho.results]
    scatter(times,
            error,
            label = "",
            xaxis = "Time per force per atom | µs",
            yaxis = "we MSE(E, E') + wf MSE(F, F')")
end


# Hyperparamter optimization ###################################################
e_mae_max = 0.05
f_mae_max = 0.05
weights = [1.0, 1.0]
intercept = true

ho = Hyperoptimizer(10,
                    species           = [[:Hf, :O]],
                    body_order        = [2, 3, 4],
                    polynomial_degree = [3, 4, 5],
                    rcutoff           = [4.5, 5.0, 5.5],
                    wL                = [0.5, 1.0, 1.5],
                    csp               = [0.5, 1.0, 1.5],
                    r0                = [0.5, 1.0, 1.5])

for (i, species, body_order, polynomial_degree, rcutoff, wL, csp, r0) in ho
    basis = ACE(species           = species,
                body_order        = body_order,
                polynomial_degree = polynomial_degree,
                rcutoff           = rcutoff,
                wL                = wL,
                csp               = csp,
                r0                = r0)
    iap = LBasisPotential(basis)
    e_descr_new = compute_local_descriptors(conf_train, iap.basis, pbar = false)
    f_descr_new = compute_force_descriptors(conf_train, iap.basis, pbar = false)
    ds_cur = DataSet(conf_train .+ e_descr_new .+ f_descr_new)
    learn!(iap, ds_cur, weights, intercept)
    e, e_pred = get_all_energies(ds_cur), get_all_energies(ds_cur, iap)
    f, f_pred = get_all_forces(ds_cur), get_all_forces(ds_cur, iap)
    e_mae, e_rmse, e_rsq = calc_metrics(e_pred, e)
    f_mae, f_rmse, f_rsq = calc_metrics(f_pred, f)
    time_us  = estimate_time(conf_train, iap) * 10^6
    error = weights[1] * e_rmse^2 + weights[2] * f_rmse^2
    metrics  = OrderedDict( :error     => error,
                            :e_mae     => e_mae,
                            :e_rmse    => e_rmse,
                            :e_rsq     => e_rsq,
                            :f_mae     => f_mae,
                            :f_rmse    => f_rmse,
                            :f_rsq     => f_rsq,
                            :time_us   => time_us)
    if e_mae < e_mae_max &&
       f_mae < f_mae_max
       loss = time_us
    else
       loss = time_us + error * 10^3
    end
    println("")
    print("E_MAE:$(round(e_mae; digits=3)), ")
    print("F_MAE:$(round(f_mae; digits=3)), ")
    println("Time per force per atom | µs:$(round(time_us; digits=3))")
    flush(stdout)
    push!(ho.history, (species, body_order, polynomial_degree, rcutoff, wL, csp, r0))
    push!(ho.results, (loss, metrics, iap))
end

# Post-process output: calculate metrics, create plots, and save results #######

# Prnt and save optimization results
results = get_results(ho)
println(results)
@save_dataframe path results

# Optimal IAP
opt_iap = ho.minimum[3]
@save_var res_path opt_iap.β
@save_var res_path opt_iap.β0
@save_var res_path opt_iap.basis

# Plot loss vs time
err_time = plot_err_time(ho)
@save_fig res_path err_time
DisplayAs.PNG(err_time)

