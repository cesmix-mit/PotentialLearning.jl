using Hyperopt
using InteratomicBasisPotentials

include("InteratomicBasisPotentialsExt.jl")
include("HyperOptExt.jl")
include("dbscan.jl")

function hyperlearn!(   model,
                        model_pars,
                        ho_pars,
                        configurations,
                        dataset_selector = Nothing,
                        dataset_generator = Nothing,
                        max_iterations = 1,
                        end_condition = true,
                        acc_threshold = 0.1,
                        weights = [1.0, 1.0], 
                        intercept = false)
    
    hyper_optimizer = inject_pars(ho_pars, model_pars, 
    quote
         @hyperopt for i = n_samples

            iap = model(state...)
            lb = LBasisPotentialExt(iap)
            
            # Dataset selection
            inds = get_random_subset(dataset_selector)
            conf_new = conf_train[inds]
            
            # Compute energy and force descriptors of new sampled configurations
            e_descr_new = compute_local_descriptors(conf_new, iap, pbar = false)
            f_descr_new = compute_force_descriptors(conf_new, iap, pbar = false)
            ds_cur = DataSet(conf_new .+ e_descr_new .+ f_descr_new)
            
            # Learn
            learn!(lb, ds_cur, weights, intercept)
            
            # Get true and predicted values
            e, e_pred = get_all_energies(ds_cur), get_all_energies(ds_cur, lb)
            f, f_pred = get_all_forces(ds_cur), get_all_forces(ds_cur, lb)
            
            # Compute metrics
            e_mae, e_rmse, e_rsq = calc_metrics(e_pred, e)
            f_mae, f_rmse, f_rsq = calc_metrics(f_pred, f)
            
            # Compute loss
            accuracy = weights[1] * e_rmse^2 + weights[2] * f_rmse^2
            ndesc = length(e_descr_new[1])
            loss = accuracy < acc_threshold ? ndesc : ndesc * accuracy
            
            # Print results
            println("Learning experiment: $i. E_MAE: $e_mae, F_MAE: $f_mae, loss: $loss.")
            
            # Return loss
            HOResult(loss, lb)
        end
    end)
    return hyper_optimizer
end

