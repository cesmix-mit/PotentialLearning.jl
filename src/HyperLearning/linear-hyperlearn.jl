create_ho(x) = Hyperoptimizer(1)

"""
function hyperloss(
    metrics::OrderedDict:
    w_e       = 1.0,
    w_f       = 1.0,
    w_t       = 1.0E-3,
    e_mae_max = 0.05,
    f_mae_max = 0.05
)

`metrics`: OrderedDict object with metrics of the fitting process. 
    - Mean absolute error of energies: e_mae.
    - Mean absolute error of forces: f_mae.
    - Time per force per atom: time_us.
`w_e`: energy weight.
`w_f`: force weight.
`w_t`: time weight.
`e_mae_max`: maximum mean absolute error for energies.
`f_mae_max`: maximum mean absolute error for forces.

Loss function for hyper-parameter optimization: minimizes fitting error and time.
"""
function hyperloss(
    metrics::OrderedDict;
    w_e       = 1.0,
    w_f       = 1.0,
    w_t       = 1.0E-3,
    e_mae_max = 0.05,
    f_mae_max = 0.05
)
    e_mae     = metrics[:e_mae]
    f_mae     = metrics[:f_mae]
    time_us   = metrics[:time_us]
    w_e = w_e * e_mae/e_mae_max
    w_f = w_f * f_mae/f_mae_max
    loss = w_e * e_mae + w_f * e_mae + w_t * time_us
    return loss
end;

"""
function hyperlearn!(
    model::DataType,
    pars::OrderedDict,
    conf_train::DataSet;
    n_samples = 5,
    sampler = RandomSampler(),
    loss = loss,
    ws = [1.0, 1.0],
    int = true
)

Hyper-parameter optimization of linear interatomic potentials.
"""
function hyperlearn!(
    model::DataType,
    pars::OrderedDict,
    conf_train::DataSet;
    n_samples = 5,
    sampler = RandomSampler(),
    loss = loss,
    ws = [1.0, 1.0],
    int = true
)
    s = "create_ho(sampler) = Hyperoptimizer($n_samples, sampler, " *
         join("$k = $v, " for (k, v) in pars) * ")"
    eval(Meta.parse(s))
    ho = Base.invokelatest(create_ho, sampler)
    if (ho.sampler isa LHSampler) || (ho.sampler isa CLHSampler)
        Hyperopt.init!(ho.sampler, ho)
    end
    species = get_species(conf_train)
    for (i, state...) in ho
        basis = model(; species = species, state...)
        iap = LBasisPotential(basis)
        ## Compute energy and force descriptors
        e_descr_new = compute_local_descriptors(conf_train, iap.basis, pbar = false)
        f_descr_new = compute_force_descriptors(conf_train, iap.basis, pbar = false)
        ds_cur = DataSet(conf_train .+ e_descr_new .+ f_descr_new)
        ## Learn
        learn!(iap, ds_cur, ws, int)
        ## Get true and predicted values
        e, e_pred = get_all_energies(ds_cur), get_all_energies(ds_cur, iap)
        f, f_pred = get_all_forces(ds_cur), get_all_forces(ds_cur, iap)
        ## Compute metrics
        e_mae, e_rmse, e_rsq = calc_metrics(e_pred, e)
        f_mae, f_rmse, f_rsq = calc_metrics(f_pred, f)
        time_us  = estimate_time(conf_train, iap) * 10^6
        metrics  = OrderedDict( :e_mae     => e_mae,
                                :e_rmse    => e_rmse,
                                :e_rsq     => e_rsq,
                                :f_mae     => f_mae,
                                :f_rmse    => f_rmse,
                                :f_rsq     => f_rsq,
                                :time_us   => time_us)
        ## Compute multi-objetive loss based on error and time
        l = hyperloss(metrics)
        ## Print results
        print("E_MAE:$(round(e_mae; digits=3)) eV/atom, ")
        print("F_MAE:$(round(f_mae; digits=3)) eV/Å, ")
        println("Time per force per atom:$(round(time_us; digits=3)) µs")
        flush(stdout)
        ## Return loss
        push!(ho.history, [v for v in state])
        push!(ho.results, (l, metrics, iap))
    end
    iap = ho.minimum[3]
    res = get_results(ho)
    return iap, res
end 

