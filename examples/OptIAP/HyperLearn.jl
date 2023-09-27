import Base: <, isless, ==, isequal, isinf
export <, isless, ==, isequal, isinf

using Hyperopt
using InteratomicBasisPotentials

include("dbscan.jl")

# Add following function to IBS.jl
function InteratomicBasisPotentials.ACE(species, body_order, polynomial_degree,
                                        wL, csp, r0, rcutoff)
    return ACE(species = species, body_order = body_order,
               polynomial_degree = polynomial_degree, 
               wL = wL, csp = csp, r0 = r0, rcutoff = rcutoff)
end

exx = nothing
function inject_pars(ho_pars, model_pars, ex)
    
    global exx = ex
    # Inject hyper-optimizer parameteres
    e = ex.args[2].args[3]
    e.args[1].args = []
    for (k,v) in eval(ho_pars)
        push!(e.args[1].args, :($k = $v))
    end
    
    # Inject model parameteres
    for (k,v) in eval(model_pars)
        push!(e.args[1].args, :($k = $v))
    end
    
    # Inject state variable
    state = [k for k in keys(eval(model_pars))]
    state_cond = Meta.parse(string("if state == nothing state = [", ["$k," for k in state]..., "] end"))
    pushfirst!(e.args[2].args, state_cond)
    
    # Inject state variable in return tuple, only when Hyperband is used
    if typeof(eval(ho_pars)[:sampler]) == Hyperband
        ret_e = e.args[2].args[end-1]
        e.args[2].args[end-1] = Meta.parse(string("$ret_e, state"))
    end

    # Evaluate new code
    eval(ex)
end

struct HOResult <: Number
    loss
    opt_iap
end

<(x::HOResult, y::HOResult) = x.loss < y.loss
<(x::Number, y::HOResult) = x < y.loss
<(x::HOResult, y::Number) = x.loss < y
isless(x::HOResult, y::HOResult) = x.loss < y.loss
isless(x::Number, y::HOResult) = x < y.loss
isless(x::HOResult, y::Number) = x.loss < y
==(x::HOResult, y::HOResult) = x.loss == y.loss
==(x::Number, y::HOResult) = x == y.loss
==(x::HOResult, y::Number) = x.loss == y
isequal(x::HOResult, y::HOResult) = x.loss == y.loss
isequal(x::Number, y::HOResult) = x == y.loss
isequal(x::HOResult, y::Number) = x.loss == y
Real(x::HOResult) = x.loss
isinf(x::HOResult) = isinf(x.loss)

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
    
    #iaps = Atomic{Int}
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
            
            # Save trained IAP
            #global iaps
            #push!(iaps, state => lb)
            
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
    return hyper_optimizer#, iaps[hyper_optimizer.minimum]
end

