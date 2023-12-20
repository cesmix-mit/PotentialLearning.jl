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


function train_potential(ds_train::DataSet, ace::ACE, dpp_batch::Int64)
    # learn with DPP
    lb = LBasisPotential(ace)
    dpp = kDPP(ds_train, GlobalMean(), DotProduct(); batch_size = dpp_batch)
    dpp_inds = get_random_subset(dpp)
    lp = learn!(lb, ds_train[dpp_inds], [100, 1], false)

    cond_num = compute_cond_num(lp)
    return lb, cond_num, dpp_inds
end 


function compute_cond_num(lp::PotentialLearning.UnivariateLinearProblem)
    A = reduce(hcat, lp.iv_data)'
    return cond(A)
end


function compute_cond_num(lp::PotentialLearning.CovariateLinearProblem)
    B = reduce(hcat, lp.B)'
    dB = reduce(hcat, lp.dB)'
    A = [B; dB]
    return cond(A)
end


function cross_validation_training(ds, ace; ndiv=10,
    dpp_batch=Int(floor(2*length(ds) / ndiv))
    )

    # retrieve DFT data
    energies = get_all_energies(ds)
    forces = reduce(vcat,[sum(norm.(get_values(get_forces(ds[c]))))
                for c in 1:length(ds)]) # magnitude
    
    # init arrays
    cond_num = zeros(ndiv)
    e_err = Matrix{Float64}(undef, (length(ds), ndiv))
    f_err = Matrix{Float64}(undef, (length(ds), ndiv))
    e_mae, e_rmse = Dict(f => zeros(ndiv) for f in file_arr), Dict(f => zeros(ndiv) for f in file_arr)
    f_mae, f_rmse = Dict(f => zeros(ndiv) for f in file_arr), Dict(f => zeros(ndiv) for f in file_arr)
    sel_ind = Vector{Vector{Int64}}(undef, ndiv)

    # make random divisions of data
    ndata = length(ds)
    ncut = Int(floor(ndata / ndiv))
    ind_all = rand(1:ndata, ndata)
    ind = [ind_all[(k*ncut+1):((k+1)*ncut)] for k = 0:(ndiv-1)]

    # iterate over divisions
    for i = 1:ndiv
        println("batch $i")
        # split train/test sets
        train_ind = reduce(vcat, ind[Not(i)])
        ds_train = ds[train_ind]

        # train using dpp
        lb, cond_num[i], dpp_ind = train_potential(ds_train, ace, dpp_batch)
        sel_ind[i] = train_ind[dpp_ind]

        # get predicted energies
        e_pred = get_all_energies(ds, lb)
        f_pred = get_all_forces_mag(ds, lb)
        e_err[:,i] = energies - e_pred 
        f_err[:,i] = forces - f_pred

        for j = 1:nfile
            e_mae[file_arr[j]][i], e_rmse[file_arr[j]][i], _ = calc_metrics(energies[confs_id[j]], e_pred[confs_id[j]])
            f_mae[file_arr[j]][i], f_rmse[file_arr[j]][i], _ = calc_metrics(forces[confs_id[j]], f_pred[confs_id[j]])
        end
    end

    # populate DataFrame
    df_conf = DataFrame("config" => 1:length(ds),
            "file" => reduce(vcat, [[file_arr[j] for i = 1:length(confs_arr[j])] for j = 1:nfile]),
            "DFT energy" => energies,
            "energy err mean" => mean(e_err, dims=2)[:], # mean error from k-fold CV
            "energy err std" => std(e_err, dims=2)[:], # std of error
            "DFT force" => forces,
            "force err mean" =>  mean(f_err, dims=2)[:], # mean error from k-fold CV
            "force err std" => std(f_err, dims=2)[:], # std of error
    )

    df_meta = DataFrame("file" => file_arr,
            "# configs" => [length(conf) for conf in confs_id],
            "E mae mean" => [mean(e_mae[k]) for k in keys(e_mae)],
            "E mae std" => [std(e_mae[k]) for k in keys(e_mae)],
            "E rmse mean" => [mean(e_rmse[k]) for k in keys(e_rmse)],
            "E rmse std" => [std(e_rmse[k]) for k in keys(e_rmse)],
            "F mae mean" => [mean(f_mae[k]) for k in keys(f_mae)],
            "F mae std" => [std(f_mae[k]) for k in keys(f_mae)],
            "F rmse mean" => [mean(f_rmse[k]) for k in keys(f_rmse)],
            "F rmse std" => [std(f_rmse[k]) for k in keys(f_rmse)],
    )

    # write to file
    CSV.write(outpath*"$(elname)_ACE-$(nbody)-$(deg)_train_full_nDPP=$(dpp_batch).csv", df_conf)
    CSV.write(outpath*"$(elname)_ACE-$(nbody)-$(deg)_train_metadata_nDPP=$(dpp_batch).csv", df_meta)

    return sel_ind, cond_num
end


# function get_all_forces_mag(ds::DataSet)
    #     for c in ds
    #         force_coord = reduce(hcat, get_values(get_forces(c)))'
    #         sum(force_coord, dims=1)



function get_all_forces_mag(
    ds::DataSet,
    lb::PotentialLearning.LinearBasisPotential
)
    force_descriptors = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds]
    force_pred = [lb.β0[1] .+  dB' * lb.β for dB in [reduce(hcat, fi) for fi in force_descriptors]]
    return [sum(norm.([f[k:k+2] for k = 1:3:length(f)])) for f in force_pred]
end
