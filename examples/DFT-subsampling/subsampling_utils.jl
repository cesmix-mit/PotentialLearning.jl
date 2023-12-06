# master function for iteration ------------------------------------------------------------------

function DPP_training_trial(
    elem::Vector,
    nbody::Int64,
    deg::Int64,
    batch_size::Vector,
    conf::DataSet,
    save_dir::String;
    nfold::Int64=10)

    # Define directory
    exp_dir = "$(save_dir)ACE_$(nbody)body_$(deg)deg/"
    run(`mkdir -p $exp_dir`)

    # Define ACE basis
    ace = ACE(species = elem,             # species
            body_order = nbody,           # n-body
            polynomial_degree = deg,      # degree of polynomials
            wL = 1.0,                     # Defaults, See ACE.jl documentation 
            csp = 1.0,                    # Defaults, See ACE.jl documentation 
            r0 = 1.0,                     # minimum distance between atoms
            rcutoff = 10.0)

    # Update dataset by adding energy (local) descriptors
    println("Computing local descriptors")

    e_descr = JLD.load(exp_dir*"energy_descriptors_allfiles.jld")["e_descr"]
    f_descr = JLD.load(exp_dir*"force_descriptors_allfiles.jld")["f_descr"]

    # @time e_descr = compute_local_descriptors(conf, ace)
    # @time f_descr = compute_force_descriptors(conf, ace)
    # JLD.save(exp_dir*"energy_descriptors.jld", "e_descr", e_descr)
    # JLD.save(exp_dir*"force_descriptors.jld", "f_descr", f_descr)

    ds = DataSet(conf .+ e_descr .+ f_descr)

    # Compute cross validation error from training
    # res_dpp, res_srs = cross_validation_training(ds, ace, batch_size, exp_dir; nfold=nfold)
    println("Monte Carlo trials")
    res_dpp, res_srs = monte_carlo_training(ds, ace, batch_size, exp_dir; niter=nfold)

end


# helper functions for training potential ------------------------------------------------------------------

# train using DPP sampling
function train_potential(
    ds::DataSet,
    ace::ACE,
    L::EllEnsemble,
    batch_size::Int64;
    α=1e-8)

    # init basis potential
    lb = LBasisPotentialExt(ace)
    # compute kDPP
    rescale!(L, batch_size)
    dpp = kDPP(L, batch_size)
    # draw subset
    inds = get_random_subset(dpp)
    # learning problem
    lp = learn!(lb, ds[inds], α)

    return lp, lb, inds
end 


# train using simple random sampling
function train_potential(
    ds::DataSet,
    ace::ACE,
    batch_size::Int64;
    α=1e-8)

    # init basis potential
    lb = LBasisPotentialExt(ace)
    # define random selector
    srs = RandomSelector(length(ds), batch_size)
    # draw subset
    inds = get_random_subset(srs)
    # learning problem
    lp = learn!(lb, ds[inds], α)

    return lp, lb, inds
end 


function monte_carlo_training(
    ds::DataSet,
    ace::ACE,
    batch_size::Vector,
    save_dir::String;
    niter=10,
    )
    
    # init results dict
    res_srs = init_res_dict(batch_size, niter)
    res_dpp = init_res_dict(batch_size, niter)

    # split train set
    ndata = length(ds)
    ind_all = rand(1:ndata, ndata)
    cut = Int(floor(ndata * 0.5))
    train_ind = ind_all[1:cut]
    test_ind = ind_all[cut+1:end]
    ds_train = ds[train_ind]
    ds_test = ds[test_ind]

    # compute L-ensemble for DPP once
    t = @elapsed L = compute_ell_ensemble(ds_train, GlobalMean(), DotProduct())
    println("Compute L-ensemble: $t sec. ")


    # iterate over divisions
    for i = 1:niter
        println("----------- iter $i ----------")

        for bs in batch_size
            t = @elapsed begin
                # train by DPP
                lpd, lbd, dpp_ind = train_potential(ds_train, ace, L, bs)
                res_dpp[bs] = update_res_dict(i, res_dpp[bs], ds_test, lpd, lbd, dpp_ind)

                # train by simple random sampling
                lps, lbs, srs_ind = train_potential(ds_train, ace, bs)
                res_srs[bs] = update_res_dict(i, res_srs[bs], ds_test, lps, lbs, srs_ind)

                JLD.save(save_dir*"DPP_training_results_MC_N=$bs.jld", "res", res_dpp[bs])
                JLD.save(save_dir*"SRS_training_results_MC_N=$bs.jld", "res", res_srs[bs])
            end
            println("Train with batch $bs: $t sec.")
        end
    end

    JLD.save(save_dir*"DPP_training_results_MC_all.jld", "res", res_dpp)
    JLD.save(save_dir*"SRS_training_results_MC_all.jld", "res", res_srs)

    return res_dpp, res_srs
end



# cross validation training
function cross_validation_training(
    ds::DataSet,
    ace::ACE,
    batch_size::Vector,
    save_dir::String;
    ndiv=10,
    )
    
    # init results dict
    res_srs = init_res_dict(batch_size, ndiv)
    res_dpp = init_res_dict(batch_size, ndiv)

    # make random divisions of data
    ndata = length(ds)
    ncut = Int(floor(ndata / ndiv))
    ind_all = rand(1:ndata, ndata)
    cv_ind = [ind_all[(k*ncut+1):((k+1)*ncut)] for k = 0:(ndiv-1)]

    # iterate over divisions
    for i = 1:ndiv
        println("----------- fold $i ----------")

        # split train set
        train_ind = reduce(vcat, cv_ind[Not(i)]) # leave out one cut
        test_ind = cv_ind[i]
        ds_train = ds[train_ind]
        ds_test = ds[test_ind]

        # compute L-ensemble for DPP once
        t = @elapsed L = compute_ell_ensemble(ds_train, GlobalMean(), DotProduct())
        println("Compute L-ensemble: $t sec. ")

        for bs in batch_size
            t = @elapsed begin
                # train by DPP
                lpd, lbd, dpp_ind = train_potential(ds_train, ace, L, bs)
                res_dpp[bs] = update_res_dict(i, res_dpp[bs], ds_test, lpd, lbd, dpp_ind)

                # train by simple random sampling
                lps, lbs, srs_ind = train_potential(ds_train, ace, bs)
                res_srs[bs] = update_res_dict(i, res_srs[bs], ds_test, lps, lbs, srs_ind)

                JLD.save(save_dir*"DPP_training_results_N=$bs.jld", "res", res_dpp[bs])
                JLD.save(save_dir*"SRS_training_results_N=$bs.jld", "res", res_srs[bs])
            end
            println("Train with batch $bs: $t sec.")
        end
    end

    JLD.save(save_dir*"DPP_training_results_all.jld", "res", res_dpp)
    JLD.save(save_dir*"SRS_training_results_all.jld", "res", res_srs)

    return res_dpp, res_srs
end


# compute condition number ------------------------------------------------------------------

# for UnivariateLinearProblem
function compute_cond_num(lp::PotentialLearning.UnivariateLinearProblem)
    A = reduce(hcat, lp.iv_data)'
    return cond(A)
end

# for CovariateLinearProblem
function compute_cond_num(lp::PotentialLearning.CovariateLinearProblem)
    B = reduce(hcat, lp.B)'
    dB = reduce(hcat, lp.dB)'
    A = [B; dB]
    return cond(A)
end





# get all force magnitudes ---------------------------------------------------------------------------

# ground truth forces
function get_all_forces_mag(ds::DataSet)
    return reduce(vcat, [norm.(get_values(get_forces(ds[c]))) for c in 1:length(ds)])
end

# predicted forces
function get_all_forces_mag(
    ds::DataSet,
    lb::PotentialLearning.LinearBasisPotential
)
    force_descriptors = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds]
    force_pred = [lb.β0[1] .+  dB' * lb.β for dB in [reduce(hcat, fi) for fi in force_descriptors]]
    return reduce(vcat, [norm.([f[k:k+2] for k = 1:3:length(f)]) for f in force_pred])
end



# record results -------------------------------------------------------------------------------------

function init_res_dict(batch_size::Vector, ndiv::Int64)
    res = Dict{Int64, Dict}(bs => Dict{String, Vector}() for bs in batch_size)
    for bs in batch_size
        res[bs]["cond_num"] = zeros(ndiv)
        # res[bs]["indices"] = Vector{Vector{Float64}}(undef, ndiv)
        # res[bs]["energy_err"] = Vector{Vector{Float64}}(undef, ndiv)
        res[bs]["energy_mae"] = zeros(ndiv)
        res[bs]["energy_rmse"] = zeros(ndiv)
        res[bs]["energy_rsq"] = zeros(ndiv)
        # res[bs]["force_err"] = Vector{Vector{Float64}}(undef, ndiv)
        res[bs]["force_mae"] = zeros(ndiv)
        res[bs]["force_rmse"] = zeros(ndiv)
        res[bs]["force_rsq"] = zeros(ndiv)
    end
    return res
end


function update_res_dict(
    i::Int64,
    res::Dict,
    ds::DataSet,
    lp,
    lb::LBasisPotentialExt,
    ind::Vector, 
)
    res["cond_num"][i] = compute_cond_num(lp)
    # res["indices"][i] = ind

    # get DFT and predicted energies/forces
    energies = get_all_energies(ds)
    forces = get_all_forces_mag(ds) # magnitude
    e_pred = get_all_energies(ds, lb)
    f_pred = get_all_forces_mag(ds, lb)

    # compute errors
    # res["energy_err"][i] = energies - e_pred 
    # res["force_err"][i] = forces - f_pred
    res["energy_mae"][i], res["energy_rmse"][i], res["energy_rsq"][i] = calc_metrics(energies, e_pred)
    res["force_mae"][i], res["force_rmse"][i], res["force_rsq"][i] = calc_metrics(forces, f_pred)
    
    return res
end


function compute_cv_metadata(
    res::Dict,
)
    batches = sort(collect(keys(res)))

    df_conf = DataFrame(
        "batch size" => batches,
        # "DFT energy" => get_all_energies(ds),
        # "E err mean" => mean(res["energy_err"]), # mean error from k-fold CV
        # "E err std" => std(res["energy_err"]), # std of error
        "E mae min" => [minimum(res[bs]["energy_mae"]) for bs in batches],
        "E mae med" => [median(res[bs]["energy_mae"]) for bs in batches],
        "E mae lqt" => [quantile!(res[bs]["energy_mae"], 0.25) for bs in batches],
        "E mae uqt" => [quantile!(res[bs]["energy_mae"], 0.75) for bs in batches],

        "E rmse min" => [minimum(res[bs]["energy_rmse"]) for bs in batches],
        "E rmse med" => [median(res[bs]["energy_rmse"]) for bs in batches],
        "E rmse lqt" => [quantile!(res[bs]["energy_rmse"], 0.25) for bs in batches],
        "E rmse uqt" => [quantile!(res[bs]["energy_rmse"], 0.75) for bs in batches],
        
        "E rsq max" => [maximum(abs.(res[bs]["energy_rsq"])) for bs in batches],
        "E rsq med" => [median(abs.(res[bs]["energy_rsq"])) for bs in batches],
        "E rsq lqt" => [quantile!(abs.(res[bs]["energy_rsq"]), 0.25) for bs in batches],
        "E rsq uqt" => [quantile!(abs.(res[bs]["energy_rsq"]), 0.75) for bs in batches],

        # "DFT force" => get_all_forces_mag(ds),
        # "F err mean" =>  mean(res["force_err"]), # mean error from k-fold CV
        # "F err std" => std(res["force_err"]), # std of error
        "F mae min" => [minimum(res[bs]["force_mae"]) for bs in batches],
        "F mae med" => [median(res[bs]["force_mae"]) for bs in batches],
        "F mae lqt" => [quantile!(res[bs]["force_mae"], 0.25) for bs in batches],
        "F mae uqt" => [quantile!(res[bs]["force_mae"], 0.75) for bs in batches],

        "F rmse min" => [minimum(res[bs]["force_rmse"]) for bs in batches],
        "F rmse med" => [median(res[bs]["force_rmse"]) for bs in batches],
        "F rmse lqt" => [quantile!(res[bs]["force_rmse"], 0.25) for bs in batches],
        "F rmse uqt" => [quantile!(res[bs]["force_rmse"], 0.75) for bs in batches],

        "F rsq max" => [maximum(abs.(res[bs]["force_rsq"])) for bs in batches],
        "F rsq med" => [median(abs.(res[bs]["force_rsq"])) for bs in batches],
        "F rsq lqt" => [quantile!(abs.(res[bs]["force_rsq"]), 0.25) for bs in batches],
        "F rsq uqt" => [quantile!(abs.(res[bs]["force_rsq"]), 0.75) for bs in batches],
    )
    return df_conf
end







