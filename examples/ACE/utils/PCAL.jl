# A parallel cluster-based active learning (PCAL) method of IAPs parameters


# Parallel cluster-based active learning problem ###############################
struct PCALProblem
    iap
    e_mae_tol
    f_mae_tol
    n_clusters
    sample_size
    w_e
    w_f
end

function PCALProblem(iap; e_mae_tol = 0.2, f_mae_tol = 0.2,
                     n_clusters = 10, sample_size = 10, w_e = 1, w_f = 1)
    return PCALProblem(iap, e_mae_tol, f_mae_tol, n_clusters, sample_size, w_e, w_f)
end


# Parallel cluster-based active learning of IAP parameters #####################
function learn!(pcal::PCALProblem, ds::DataSet)

    # Iteratively increase training dataset and fit IAP until reach threasholds
    println("Starting parallel cluster-based active learning of IAP parameters...\n")
    lp = []; ds_cur = []; conf_cur = []
    e_descr_cur = []; f_descr_cur = []
    e_mae, e_rmse, e_rsq = Inf, Inf, Inf
    f_mae, f_rmse, f_rsq = Inf, Inf, Inf
    i = 1
    clusters = get_clusters(pcal.n_clusters, ds)
    while ( e_mae > pcal.e_mae_tol || f_mae > pcal.f_mae_tol ) && length(ds_cur) < length(ds)

        println("Active learning iteration: $i")

        # Select new configurations by sampling from clusters
        sample(c, n) = [c[rand(1:length(c))] for _ in 1:n]
        inds = reduce(vcat, [sample(c, pcal.sample_size) for c in clusters])
        conf_new = ds[inds]
        
        # Compute energy and force descriptors of new sampled configurations
        println("Computing energy descriptors of new sampled configurations...")
        e_descr_new = compute_local_descriptors(conf_new, pcal.iap.basis)
        println("Computing force descriptors of new sampled configurations...")
        f_descr_new = compute_force_descriptors(conf_new, pcal.iap.basis)

        # Update current configurations, energy and force descriptors, and dataset
        push!(conf_cur, conf_new...)
        push!(e_descr_cur, e_descr_new...)
        push!(f_descr_cur, f_descr_new...)
        ds_cur = DataSet(conf_cur .+ e_descr_cur .+ f_descr_cur)
        println("Current number of configurations: $(length(ds_cur))")

        # Learn IAP parameters using increased training dataset
        println("Learning energies and forces...")
        learn!(pcal.iap, ds_cur; w_e = input["w_e"], w_f = input["w_f"]) # learn!(lb, ds_train)

        # Get true and predicted values
        e, f = get_all_energies(ds_cur), get_all_forces(ds_cur)
        e_pred, f_pred = get_all_energies(ds_cur, pcal.iap), get_all_forces(ds_cur, pcal.iap)
        
        # Compute metrics
        e_mae, e_rmse, e_rsq = calc_metrics(e_pred, e)
        f_mae, f_rmse, f_rsq = calc_metrics(f_pred, f)
        println("e_mae: $e_mae, e_rmse: $e_rmse, e_rsq: $e_rsq")
        println("f_mae: $f_mae, f_rmse: $f_rmse, f_rsq: $f_rsq \n")
        
        # Update iteration number
        i += 1
        
        GC.gc()
    end
    
    copyto!(ds.Configurations, ds_cur.Configurations)

    println("Active learning process completed.\n")

end

# Calculate clusters of dataset ################################################
function get_clusters(ds; eps = 0.05, minpts = 10)
    # Create distance matrix
    n = length(ds); d = zeros(n, n)
    for i in 1:n
        p1 = Matrix(hcat(get_values.(get_positions(ds[i]))...)')
        for j in i+1:n
            p2 = Matrix(hcat(get_values.(get_positions(ds[j]))...)')
            d[i,j] = kabsch_rmsd(p1, p2)
            d[j,i] = d[i,j]
        end
    end
    d = Symmetric(d)
    # Create clusters using dbscan
    c = dbscan(d, eps, minpts)
    a = c.assignments # get the assignments of points to clusters
    clusters = [findall(x->x==i, a) for i in 1:n_clusters]
    return clusters
end


