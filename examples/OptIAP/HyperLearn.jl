using Hyperopt
using InteratomicBasisPotentials

include("InteratomicBasisPotentialsExt.jl")
include("HyperOptExt.jl")
include("dbscan.jl")


function hyperlearn!(   conf_train,
                        model,
                        model_pars,
                        ho_pars;
                        dataset_selector = Nothing,
                        dataset_generator = Nothing,
                        acc_threshold = 0.1,
                        max_iterations = 1,
                        end_condition = true,
                        weights = [1.0, 1.0], 
                        intercept = false)
    
    hyper_optimizer = inject_pars(ho_pars, model_pars, 
    quote
         @hyperopt for i = n_samples
            basis = model(state...)
            iap = LBasisPotentialExt(basis)
            
            # Compute energy and force descriptors
            e_descr_new = compute_local_descriptors(conf_train, iap.basis, pbar = false)
            f_descr_new = compute_force_descriptors(conf_train, iap.basis, pbar = false)
            ds_cur = DataSet(conf_train .+ e_descr_new .+ f_descr_new)
            
            # Learn
            learn!(iap, ds_cur, weights, intercept)
            
            # Get true and predicted values
            e, e_pred = get_all_energies(ds_cur), get_all_energies(ds_cur, iap)
            f, f_pred = get_all_forces(ds_cur), get_all_forces(ds_cur, iap)
            
            # Compute metrics
            e_mae, e_rmse, e_rsq = calc_metrics(e_pred, e)
            f_mae, f_rmse, f_rsq = calc_metrics(f_pred, f)

            # Compute accuracy based on MSE of energies and forces
            accuracy = weights[1] * e_rmse^2 + weights[2] * f_rmse^2

            # Estimate time to compute forces [ms]
            time = estimate_time(conf_train, iap) * 1000.0
            
            # Compute loss based on accuracy and time
            loss = accuracy < acc_threshold ? time : time + accuracy
            
            # Print results
            print("E_MAE:$(round(e_mae; digits=3)), ")
            print("F_MAE:$(round(f_mae; digits=3)), ")
            println("Time per force per atom [ms]:$(round(time; digits=6))")
            
            # Return loss
            HOResult(loss, accuracy, time, iap)
        end
    end)
    return hyper_optimizer
end


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


function plot_loss_time(hyper_optimizer)
    accuracies = map(x -> x.accuracy,
                 hyper_optimizer.results)
    times = map(x -> x.time,
                hyper_optimizer.results)
    scatter(times,
            accuracies,
            label = "",
            xaxis = "Time | s",
            yaxis = "Accuracy")
end


