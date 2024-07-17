# Fit ACE and postprocess results. Used in subsampling experiments #############

# ## Setup experiment

# Load packages.
using AtomsBase, InteratomicPotentials, PotentialLearning
using Unitful, UnitfulAtomic
using LinearAlgebra, Random, DisplayAs
using DataFrames, Plots

# Define paths.
base_path = haskey(ENV, "BASE_PATH") ? ENV["BASE_PATH"] : "../../"
ds_path   = "$base_path/examples/data/Hf/"
res_path  = "$base_path/examples/Parallel-DPP-ACE-HfO2/results/";

# Load utility functions.
include("$base_path/examples/utils/utils.jl")


function fit(path, ds_train, ds_test, basis)

    # Learn
    lb = LBasisPotential(basis)
    ws, int = [1.0, 1.0], false
    learn!(lb, ds_train, ws, int)

    @save_var path lb.β
    @save_var path lb.β0

    # Post-process output: calculate metrics, create plots, and save results #######

    # Get true and predicted values
    n_atoms_train = length.(get_system.(ds_train))
    n_atoms_test = length.(get_system.(ds_test))

    e_train, e_train_pred = get_all_energies(ds_train) ./ n_atoms_train,
                            get_all_energies(ds_train, lb) ./ n_atoms_train
    f_train, f_train_pred = get_all_forces(ds_train),
                            get_all_forces(ds_train, lb)
    @save_var path e_train
    @save_var path e_train_pred
    @save_var path f_train
    @save_var path f_train_pred

    e_test, e_test_pred = get_all_energies(ds_test) ./ n_atoms_test,
                          get_all_energies(ds_test, lb) ./ n_atoms_test
    f_test, f_test_pred = get_all_forces(ds_test),
                          get_all_forces(ds_test, lb)
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

    f_plot = plot_forces(f_train, f_train_pred,
                         f_test, f_test_pred)
    @save_fig path f_plot

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
    
    return e_train_metrics, f_train_metrics, 
           e_test_metrics, f_test_metrics
end


# Load training and test configuration datasets ################################

paths = [
#         "$ds_path/Hf2_gas_form_sorted.extxyz", # ERROR: LoadError: SingularException(18)
#         "$ds_path/Hf2_mp103_EOS_1D_form_sorted.extxyz", # 200, :)
#         "$ds_path/Hf2_mp103_EOS_3D_form_sorted.extxyz", # 9377, :(
         "$ds_path/Hf2_mp103_EOS_6D_form_sorted.extxyz", # 17.2k, :-D or out of memory
#         "$ds_path/Hf128_MC_rattled_mp100_form_sorted.extxyz", # 306, :(
#         "$ds_path/Hf128_MC_rattled_mp103_form_sorted.extxyz", # 50, ...
#         "$ds_path/Hf128_MC_rattled_random_form_sorted.extxyz", # 498, :(
#         "$ds_path/Hf_mp100_EOS_1D_form_sorted.extxyz", # 201, ??
#         "$ds_path/Hf_mp100_primitive_EOS_1D_form_sorted.extxyz"
         ]

confs = []
for ds_path in paths
    push!(confs, load_data(ds_path, uparse("eV"), uparse("Å"))...)
end
confs = DataSet(confs)
n = length(confs)
GC.gc()

#ds_path = string("../data/HfO2_large/HfO2_figshare_form_sorted.extxyz")
#confs = load_data(ds_path, uparse("eV"), uparse("Å"))
#n = length(confs)

run(`mkdir -p $res_path`);

species = unique(vcat([atomic_symbol.(get_system(c).particles)
          for c in confs]...))

# Compute descriptors ##########################################################

# Compute ACE descriptors
basis = ACE(species           = species,
            body_order        = 4,
            polynomial_degree = 5,
            rcutoff           = 10.0,
            wL                = 1.0,
            csp               = 1.0,
            r0                = 1.0)
@save_var res_path basis

# Update training dataset by adding energy and force descriptors
println("Computing energy descriptors of dataset...")
B_time = @elapsed e_descr = compute_local_descriptors(confs, basis)
println("Computing force descriptors of dataset...")
dB_time = @elapsed f_descr = compute_force_descriptors(confs, basis)
GC.gc()
ds = DataSet(confs .+ e_descr .+ f_descr)

# Create metric dataframe
metric_names = [:exp_number,  :method, :batch_size_prop, :batch_size, :time,
                :e_train_mae, :e_train_rmse, :e_train_rsq,
                :f_train_mae, :f_train_rmse, :f_train_rsq, :f_train_mean_cos,
                :e_test_mae,  :e_test_rmse,  :e_test_rsq, 
                :f_test_mae,  :f_test_rmse,  :f_test_rsq,  :f_test_mean_cos]
metrics = DataFrame([Any[] for _ in 1:length(metric_names)], metric_names)

# Subsampling experiments: subsample full dataset vs subsample dataset by chunks
n_experiments = 30 # 100
for j in 1:n_experiments
    global metrics
    
    # Define randomized training and test dataset
    n_train = 2400 #floor(Int, 0.8 * n)
    n_test = 600 #n - n_train
    rnd_inds = randperm(n)
    rnd_inds_train = rnd_inds[1:n_train]
    rnd_inds_test = rnd_inds[n_train+1:n_train+n_test] # rnd_inds[n_train+1:end]
    ds_train_rnd = @views ds[rnd_inds_train]
    ds_test_rnd  = @views ds[rnd_inds_test]

    # Subsampling experiments:  different sample sizes
    for batch_size_prop in [0.01, 0.02, 0.04, 0.08, 0.16, 0.32] #[0.05, 0.10, 0.25]
            #[0.01, 0.02, 0.04, 0.08, 0.16, 0.32] #[0.05, 0.25, 0.5, 0.75, 0.95] #[0.05, 0.10, 0.20, 0.30] #[0.05, 0.25, 0.5, 0.75, 0.95]
    
            # Experiment j - SRS ###############################################
            println("Experiment:$j, method:SRS, batch_size_prop:$batch_size_prop")
            exp_path = "$res_path/$j-SRS-bsp$batch_size_prop/"
            run(`mkdir -p $exp_path`)
            batch_size = floor(Int, n_train * batch_size_prop)
            sampling_time = @elapsed begin
                inds = randperm(n_train)[1:batch_size]
            end
            metrics_j = fit(exp_path, (@views ds_train_rnd[inds]), ds_test_rnd, basis)
            metrics_j = merge(OrderedDict("exp_number" => j,
                                          "method" => "SRS",
                                          "batch_size_prop" => batch_size_prop,
                                          "batch_size" => batch_size,
                                          "time" => sampling_time),
                              merge(metrics_j...))
            push!(metrics, metrics_j)
            @save_dataframe(res_path, metrics)

            # Experiment j - DPP ###############################################
            println("Experiment:$j, method:DPP, batch_size_prop:$batch_size_prop")
            exp_path = "$res_path/$j-DPP-bsp$batch_size_prop/"
            run(`mkdir -p $exp_path`)
            batch_size = floor(Int, n_train * batch_size_prop)
            sampling_time = @elapsed begin
                dataset_selector = kDPP(  ds_train_rnd,
                                          GlobalMean(),
                                          DotProduct();
                                          batch_size = batch_size)
                inds = get_random_subset(dataset_selector)
            end
            metrics_j = fit(exp_path, (@views ds_train_rnd[inds]), ds_test_rnd, basis)
            metrics_j = merge(OrderedDict("exp_number" => j,
                                          "method" => "DPP",
                                          "batch_size_prop" => batch_size_prop,
                                          "batch_size" => batch_size,
                                          "time" => sampling_time),
                              merge(metrics_j...))
            push!(metrics, metrics_j)
            @save_dataframe(res_path, metrics)
            
            # Experiment j - DPP′ using n_chunks ##############################
            for n_chunks in [2, 4, 8]
                println("Experiment:$j, method:DPP′(n=$n_chunks), batch_size_prop:$batch_size_prop")
                exp_path = "$res_path/$j-DPP′-bsp$batch_size_prop-n$n_chunks/"
                run(`mkdir -p $exp_path`)
                inds = Int[]
                n_chunk = n_train ÷ n_chunks
                batch_size_chunk = floor(Int, n_chunk * batch_size_prop)
                if batch_size_chunk == 0 
                    batch_size_chunk = 1
                end
                
                #sampling_time = @elapsed @threads for i in 1:n_threads
                sampling_time = @elapsed for i in 1:n_chunks
                    a, b = 1 + (i-1) * n_chunk, i * n_chunk
                    dataset_selector = kDPP(  ds_train_rnd[a:b],
                                              GlobalMean(),
                                              DotProduct();
                                              batch_size = batch_size_chunk)
                    inds_i = get_random_subset(dataset_selector)
                    append!(inds, inds_i .+ (a .- 1))
                end
                metrics_j = fit(exp_path, (@views ds_train_rnd[inds]), ds_test_rnd, basis)
                metrics_j = merge(OrderedDict("exp_number" => j,
                                              "method" => "DPP′(n:$n_chunks)",
                                              "batch_size_prop" => batch_size_prop,
                                              "batch_size" => batch_size,
                                              "time" => sampling_time),
                                  merge(metrics_j...))
                push!(metrics, metrics_j)
                @save_dataframe(res_path, metrics)
            end
            GC.gc()
    end
end

# Postprocess ##################################################################

for metric in [:e_train_mae, :f_train_mae, :e_test_mae, :f_test_mae, :time]
    scatter()
    for method in reverse(unique(metrics[:, :method])[1:end])
        batch_size_vals = metrics[metrics.method .== method, :][:, :batch_size]
        metric_vals = metrics[metrics.method .== method, :][:, metric]
        scatter!(batch_size_vals, metric_vals, label = method,
                 alpha = 0.5, dpi=300, markerstrokewidth=0, markersize=5, xaxis=:log2,
                 xlabel = "Sample size",
                 ylabel = "$metric")
    end
    savefig("$res_path/$metric-srs.png")
end

scatter()
for method in reverse(unique(metrics[:, :method])[2:end])
    batch_size_vals = metrics[metrics.method .== method, :][:, :batch_size]
    speedup_vals = metrics[metrics.method .== "DPP", :][:, :time] ./
                  metrics[metrics.method .== method, :][:, :time]
    scatter!(batch_size_vals, speedup_vals, label = "DPP time / $method time",
             alpha = 0.5, dpi=300, markerstrokewidth=0, markersize=5, xaxis=:log2,
             xlabel = "Sample size",
             ylabel = "Speedup")
end
savefig("$res_path/speedup-srs.png")


