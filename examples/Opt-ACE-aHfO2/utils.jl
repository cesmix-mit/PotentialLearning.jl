
# Estimate force calculation time
function estimate_time(confs, iap; batch_size = 50)
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

# Get results from the hyperoptimizer
function get_results(ho)
    column_names = string.(vcat(keys(ho.results[1][2])..., ho.params...))
    rows = [[values(r[2])..., p...] for (r, p) in zip(ho.results, ho.history)]
    results = DataFrame([Any[] for _ in 1:length(column_names)], column_names)
    [push!(results, r) for r in rows]
    return sort!(results)
end

# Plot fitting error vs force time (Pareto front)
function plot_err_time(res)
    e_mae = res[!, :e_mae]
    f_mae = res[!, :f_mae]
    times = res[!, :time_us]
    plot(times,
         e_mae,
         seriestype = :scatter,
         alpha = 0.55,
         thickness_scaling = 1.35,
         markersize = 3,
         markerstrokewidth = 1,
         markerstrokecolor = :black,
         markershape = :circle,
         markercolor = :gray,
         label = "MAE(E_Pred, E_DFT)")
    plot!(times,
          f_mae,
          seriestype = :scatter,
          alpha = 0.55,
          thickness_scaling = 1.35,
          markersize = 3,
          markerstrokewidth = 1,
          markerstrokecolor = :red,
          markershape = :utriangle,
          markercolor = :red2,
          label = "MAE(F_Pred, F_DFT)")
    plot!(dpi = 300,
          label = "",
          xlabel = "Time per force per atom | µs",
          ylabel = "MAE")
end


function get_species(confs)
    return unique(vcat(unique.(atomic_symbol.(get_system.(confs)))...))
end

create_ho(x) = Hyperoptimizer(1)

# hyperlearn!
function hyperlearn!(model, pars, conf_train;
                     n_samples = 5, sampler = RandomSampler(), loss = loss,
                     ws = [1.0, 1.0], int = true)

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
        err = ws[1] * e_rmse^2 + ws[2] * f_rmse^2
        metrics  = OrderedDict( :error     => err,
                                :e_mae     => e_mae,
                                :e_rmse    => e_rmse,
                                :e_rsq     => e_rsq,
                                :f_mae     => f_mae,
                                :f_rmse    => f_rmse,
                                :f_rsq     => f_rsq,
                                :time_us   => time_us)
        ## Compute multi-objetive loss based on error and time
        l = loss(metrics)
        ## Print results
        print("E_MAE:$(round(e_mae; digits=3)), ")
        print("F_MAE:$(round(f_mae; digits=3)), ")
        println("Time per force per atom | µs:$(round(time_us; digits=3))")
        flush(stdout)
        ## Return loss
        push!(ho.history, [v for v in state])
        push!(ho.results, (l, metrics, iap))
    end
    iap = ho.minimum[3]
    res = get_results(ho)
    return iap, res
end

