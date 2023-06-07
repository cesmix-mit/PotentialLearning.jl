using Clustering

# A parallel cluster-based active learning (PCAL) method of IAPs parameters
include("kabsch.jl")

# Parallel cluster-based active learning problem ###############################
struct PCALProblem
    iap
    e_mae_tol
    f_mae_tol
    sample_size
    eps
    minpts
    w_e
    w_f
end

function PCALProblem(iap; e_mae_tol = 0.2, f_mae_tol = 0.2, sample_size = 10,
                     eps = 0.05, minpts = 10, w_e = 1, w_f = 1)
    return PCALProblem(iap, e_mae_tol, f_mae_tol, sample_size,
                       eps, minpts, w_e, w_f)
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
    clusters = get_clusters(ds, eps = pcal.eps, minpts = pcal.minpts)
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

# Parallel cluster-based active learning of IAP parameters #####################
function learn2(pcal::PCALProblem, ds::DataSet, max_iter::Int)

    # Iteratively increase training dataset and fit IAP until reach threasholds
    println("Starting parallel cluster-based active learning of IAP parameters...\n")
    lp = []; ds_cur = []; conf_cur = []
    e_descr_cur = []; f_descr_cur = []
    e_mae, e_rmse, e_rsq = Inf, Inf, Inf
    f_mae, f_rmse, f_rsq = Inf, Inf, Inf
    i = 1
    clusters = get_clusters(ds, eps = pcal.eps, minpts = pcal.minpts)
    while ( e_mae > pcal.e_mae_tol || f_mae > pcal.f_mae_tol ) && length(ds_cur) < length(ds)  && i < max_iter+1

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
    
    #copyto!(ds.Configurations, ds_cur.Configurations)

    println("Active learning process completed.\n")
    return e_mae, e_rmse, e_rsq, f_mae, f_rmse, f_rsq
end

function periodic_rmsd(p1::Array{Float64,2}, p2::Array{Float64,2}, box_lengths::Array{Float64,1})
    n_atoms = size(p1, 1)
    distances = zeros(n_atoms)
    for i in 1:n_atoms
        d = p1[i, :] - p2[i, :]
        # If d is larger than half the box length subtract box length
        d = d .- round.(d ./ box_lengths) .* box_lengths
        distances[i] = norm(d)
    end
    return sqrt(mean(distances .^2))
end

function distance_matrix_periodic(ds::DataSet)
    n = length(ds); d = zeros(n, n)
    box = bounding_box(get_system(ds[1]))
    box_lengths = [get_values(box[i])[i] for i in 1:3]
    for i in 1:n
        if bounding_box(get_system(ds[i])) != box
            error("Periodic box must be the same for all configurations.")
        end
        pi = Matrix(hcat(get_values.(get_positions(ds[i]))...)')
        for j in i+1:n
            pj = Matrix(hcat(get_values.(get_positions(ds[j]))...)')
            d[i,j] = periodic_rmsd(pi, pj, box_lengths)
            d[j,i] = d[i,j]
        end
    end
    return d
end

function distance_matrix_kabsch(ds::DataSet)
    n = length(ds); d = zeros(n, n)
    for i in 1:n
        p1 = Matrix(hcat(get_values.(get_positions(ds[i]))...)')
        for j in i+1:n
            p2 = Matrix(hcat(get_values.(get_positions(ds[j]))...)')
            d[i,j] = kabsch_rmsd(p1, p2)
            d[j,i] = d[i,j]
        end
    end
    return d
end

# Calculate clusters of dataset ################################################
function get_clusters(ds; eps = 0.05, minpts = 10)
    # Create distance matrix
    if any(boundary_conditions(get_system(ds[1])) .== [Periodic()])
        d = Symmetric(distance_matrix_periodic(ds))
    else
        d = Symmetric(distance_matrix_kabsch(ds))
    end
    # Create clusters using dbscan
    c = dbscan(d, eps, minpts)
    a = c.assignments # get the assignments of points to clusters
    n_clusters = maximum(a)
    clusters = [findall(x->x==i, a) for i in 1:n_clusters]
    return clusters
end


# Auxiliary functions ##########################################################
PotentialLearning.get_values(v::SVector) = [v.data[1].val, v.data[2].val, v.data[3].val]


