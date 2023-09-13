using Hyperopt
using InteratomicBasisPotentials

include("dbscan.jl")

# struct IAPOpt
#     hyper_optimizer::Hyperoptimizer
#     subset_selector::SubsetSelector
#     dataset_generator::DatasetGenerator
#     max_iterations::Int64 # 1 or more
#     end_condition::Function
# end


# function IAPOpt(; hyper_optimizer = Nothing,
#                   subset_selector = Nothing,
#                   dataset_generator = Nothing,
#                   max_iterations = Nothing,
#                   end_condition = Nothing)
#     return IAPOpt(  hyper_optimizer,
#                     subset_selector,
#                     dataset_generator,
#                     max_iterations,
#                     end_condition)
# end

# Add following function to IBS.jl
function InteratomicBasisPotentials.ACE(species, body_order, polynomial_degree,
                                        wL, csp, r0, rcutoff)
    return ACE(species = species, body_order = body_order,
               polynomial_degree = polynomial_degree, 
               wL = wL, csp = csp, r0 = r0, rcutoff = rcutoff)
end

function hyperlearn!(;  hyper_optimizer,
                        model,
                        configurations,
                        subset_selector = Nothing,
                        dataset_generator = Nothing,
                        max_iterations = 1,
                        end_condition = true,
                        weights = [1.0, 1.0])
    #ho = Hyperoptimizer(optiap.n_samples; optiap.params...)
    for (i, pars...) in hyper_optimizer
        iap = eval(model(pars...))
        lb = LBasisPotentialExt(iap)
        
        inds = get_random_subset(subset_selector)
        conf_new = conf_train[inds]
        
        # Compute energy and force descriptors of new sampled configurations
        e_descr_new = compute_local_descriptors(conf_new, iap, pbar = false)
        f_descr_new = compute_force_descriptors(conf_new, iap, pbar = false)
        ds_cur = DataSet(conf_new .+ e_descr_new .+ f_descr_new)
        
        # Learn
        learn!(lb, ds_cur, ws, true)
        
        # Get true and predicted values
        e, f = get_all_energies(ds_cur), get_all_forces(ds_cur)
        e_pred, f_pred = get_all_energies(ds_cur, lb), get_all_forces(ds_cur, lb)
        
        # Compute metrics
        e_mae, e_rmse, e_rsq = calc_metrics(e_pred, e)
        f_mae, f_rmse, f_rsq = calc_metrics(f_pred, f)
        println("Learning experiment: $i. E_MAE: $e_mae, F_MAE: $f_mae.")
        
        # Return value
        push!(ho.results, e_mae)
        
    end
    #iap = eval(optiap.model(ho.minimizer...))
    return ho
end



## Parallel cluster-based active learning of IAP parameters #####################
#function learn!(lb::LBasisPotential, ds, s, w_e, w_f)

#    # Iteratively increase training dataset and fit IAP until reach threasholds
#    println("Starting parallel cluster-based active learning of IAP parameters...\n")
#    lp = []; ds_cur = []; conf_cur = []
#    e_descr_cur = []; f_descr_cur = []
#    e_mae, e_rmse, e_rsq = Inf, Inf, Inf
#    f_mae, f_rmse, f_rsq = Inf, Inf, Inf
#    e_mae_tol, f_mae_tol = 0.3, 0.3
#    max_iter = 10
#    batch_size = 10
#    i = 1
#    while i <= max_iter && length(ds_cur) < length(ds) && 
#         (e_mae > e_mae_tol || f_mae > f_mae_tol)

#        println("Active learning iteration: $i")

#        # Select new configurations by sampling from clusters
#        inds = get_random_subset(s, batch_size)
#        conf_new = ds[inds]

#        # Compute energy and force descriptors of new sampled configurations
#        println("Computing energy descriptors of new sampled configurations...")
#        e_descr_new = compute_local_descriptors(conf_new, pcal.iap.basis)
#        println("Computing force descriptors of new sampled configurations...")
#        f_descr_new = compute_force_descriptors(conf_new, pcal.iap.basis)

#        # Update current configurations, energy and force descriptors, and dataset
#        push!(conf_cur, conf_new...)
#        push!(e_descr_cur, e_descr_new...)
#        push!(f_descr_cur, f_descr_new...)
#        ds_cur = DataSet(conf_cur .+ e_descr_cur .+ f_descr_cur)
#        println("Current number of configurations: $(length(ds_cur))")

#        # Learn IAP parameters using increased training dataset
#        println("Learning energies and forces...")
#        learn!(pcal.iap, ds_cur; w_e = input["w_e"], w_f = input["w_f"]) # learn!(lb, ds_train)

#        # Get true and predicted values
#        e, f = get_all_energies(ds_cur), get_all_forces(ds_cur)
#        e_pred, f_pred = get_all_energies(ds_cur, pcal.iap), get_all_forces(ds_cur, pcal.iap)
#        
#        # Compute metrics
#        e_mae, e_rmse, e_rsq = calc_metrics(e_pred, e)
#        f_mae, f_rmse, f_rsq = calc_metrics(f_pred, f)
#        println("e_mae: $e_mae, e_rmse: $e_rmse, e_rsq: $e_rsq")
#        println("f_mae: $f_mae, f_rmse: $f_rmse, f_rsq: $f_rsq \n")
#        
#        # Update iteration number
#        i += 1
#        
#        GC.gc()
#    end
#    
#    #copyto!(ds.Configurations, ds_cur.Configurations)

#    println("Active learning process completed.\n")
#    
#    return e_mae, e_rmse, e_rsq, f_mae, f_rmse, f_rsq

#end

