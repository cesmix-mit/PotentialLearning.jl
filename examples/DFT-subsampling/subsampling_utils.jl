# master function for iteration ------------------------------------------------------------------

function DPP_training_trial(
    elem::Vector,
    confs::DataSet,
    nbody::Int64,
    deg::Int64,
    batch_size::Vector,
    files::Vector{String},
    confs::Vector{DataSet},
    save_dir::String;
    nfold::Int64=10)

    # Define directory
    exp_dir = "$(save_dir)/ACE_$(nbody)body_$(deg)deg/"
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
    @time e_descr = compute_local_descriptors(confs, ace)
    @time f_descr = compute_force_descriptors(confs, ace)
    JLD.save(exp_dir*"energy_descriptors.jld", "e_descr", e_descr)
    JLD.save(exp_dir*"force_descriptors.jld", "f_descr", f_descr)

    # e_descr = JLD.load(outpath*"$(elname)_energy_descriptors.jld")["e_descr"]
    # f_descr = JLD.load(outpath*"$(elname)_force_descriptors.jld")["f_descr"]

    ds = DataSet(confs .+ e_descr .+ f_descr)
    ndata = length(ds)

    # Compute cross validation error from training
    res = cross_validation_training(ds, batch_size, exp_dir; ndiv=nfold)
    df = compute_cv_metadata(ds, res, files, confs, exp_dir)

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


# cross validation training
function cross_validation_training(
    ds::DataSet,
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
    ind = [ind_all[(k*ncut+1):((k+1)*ncut)] for k = 0:(ndiv-1)]

    # iterate over divisions
    for i = 1:ndiv
        println("batch $i")

        # split train set
        train_ind = reduce(vcat, ind[Not(i)]) # leave out one cut
        ds_train = ds[train_ind]

        # compute L-ensemble for DPP once
        L = compute_ell_ensemble(ds_train, GlobalMean(), DotProduct())

        for bs in batch_size
            # train by DPP
            lpd ,lbd, dpp_ind = train_potential(ds_train, ace, L, bs)
            res_dpp[bs] = update_res_dict(i, res_dpp[bs], ds, lpd, lbd, dpp_ind)

            # train by simple random sampling
            lps, lbs, srs_ind = train_potential(ds_train, ace, bs)
            res_srs[bs] = update_res_dict(i, res_srs[bs], ds, lps, lbs, srs_ind)

        end
    end

    JLD.save(save_dir*"DPP_training_results.jld", "res", res_dpp)
    
    return res
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


function compute_ell_ensemble(
    ds::DataSet,
    f::Feature,
    k::Kernel;
    dt = LocalDescriptors,
)
    K = KernelMatrix(ds, f, k; dt = dt)
    ell = EllEnsemble(K)
    return ell
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
        res[bs]["indices"] = Vector{Vector{Float64}}(undef, ndiv)
        res[bs]["energy_err"] = Vector{Vector{Float64}}(undef, ndiv)
        res[bs]["energy_mae"] = zeros(ndiv)
        res[bs]["energy_rmse"] = zeros(ndiv)
        res[bs]["energy_rsq"] = zeros(ndiv)
        res[bs]["force_err"] = Vector{Vector{Float64}}(undef, ndiv)
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
    res["indices"][i] = ind

    # get DFT and predicted energies/forces
    energies = get_all_energies(ds)
    forces = get_all_forces_mag(ds) # magnitude
    e_pred = get_all_energies(ds, lb)
    f_pred = get_all_forces_mag(ds, lb)

    # compute errors
    res["energy_err"][i] = energies - e_pred 
    res["force_err"][i] = forces - f_pred
    res["energy_mae"][i], res["energy_rmse"][i], res["energy_rsq"][i] = calc_metrics(energies, e_pred)
    res["force_mae"][i], res["force_rmse"][i], res["force_rsq"][i] = calc_metrics(forces, f_pred)
    
    return res
end


function compute_cv_metadata(
    ds::DataSet,
    res::Dict,
    files::Vector{String},
    confs::Vector{DataSet},
    save_dir::String
)
    batch_size = keys(res)

    for bs in batch_size
        # populate DataFrame
        df_conf = DataFrame("config" => 1:length(ds),
            "file" => reduce(vcat, [[files[j] for i = 1:length(confs[j])] for j = 1:length(confs)]),
            "DFT energy" => get_all_energies(ds),
            "E err mean" => mean(res[bs]["energy_err"]), # mean error from k-fold CV
            "E err std" => std(res[bs]["energy_err"]), # std of error
            "E mae mean" => mean(res[bs]["energy_mae"]),
            "E rmse mean" => mean(res[bs]["energy_rmse"]),
            "E rsq mean" => mean(res[bs]["energy_rsq"]),
            "DFT force" => get_all_forces_mag(ds),
            "F err mean" =>  mean(res[bs]["force_err"]), # mean error from k-fold CV
            "F err std" => std(res[bs]["force_err"]), # std of error
            "F mae mean" => mean(res[bs]["force_mae"]),
            "F rmse mean" => mean(res[bs]["force_rmse"]),
            "F rsq mean" => mean(res[bs]["force_rsq"]),
        )

        CSV.write(save_dir*"training_metadata.csv", df_conf)

        return df_conf
    end
end

# write to file
# CSV.write(outpath*"$(elname)_ACE-$(nbody)-$(deg)_train_full_nDPP=$(dpp_batch).csv", df_conf)
# CSV.write(outpath*"$(elname)_ACE-$(nbody)-$(deg)_train_metadata_nDPP=$(dpp_batch).csv", df_meta)



# I/O helper functions ----------------------------------------------------------------------
function readext(path::String, ext::String)
    dir = readdir(path)
    substr = [split(f, ".") for f in dir]
    id = findall(x -> x[end] == ext, substr)
    return dir[id]
end


function concat_dataset(confs::Vector{DataSet})
    N = length(confs)
    confs_vec = [[confs[i][j] for j = 1:length(confs[i])] for i = 1:N]
    confs_all = reduce(vcat, confs_vec)
    return DataSet(confs_all)
end