
# Data reduction algorithms ####################################################

function simple_random_sample(A, N′)
    n_train = Base.size(A, 1)
    inds = randperm(n_train)[1:N′]
    return inds
end

# kmeans-based sampling method
function kmeans_sample(A, N′)
    c = kmeans(A', 5; maxiter=200)
    a = c.assignments # get the assignments of points to clusters
    n_clusters = maximum(a)
    clusters = [findall(x->x==i, a) for i in 1:n_clusters]
    inds = reduce(vcat, Clustering.sample.(clusters, [N′ ÷ length(clusters)]))
    return inds
end

# dbscan-based sampling method
function dbscan_sample(A, N′)
    # Create clusters using dbscan
    c = dbscan(A', 10; min_neighbors = 3, min_cluster_size = 20, metric=Clustering.Euclidean())
    a = c.assignments # get the assignments of points to clusters
    n_clusters = maximum(a)
    clusters = [findall(x->x==i, a) for i in 1:n_clusters]
    inds = reduce(vcat, Clustering.sample.(clusters, [N′ ÷ length(clusters)]))
    return inds
end

# Low Rank DPP-based sampling method
function lrdpp_sample(A, N′)
    # Compute a kernel matrix for the points in x
    L = LowRank(Matrix(A))
    
    # Form an L-ensemble based on the L matrix
    dpp = EllEnsemble(L)
    
    # Sample A (obtain indices)
    _, N = Base.size(A)
    N′′ = N′ > N ? N : N′
    curr_N′ = 0
    inds = []
    while curr_N′ < N′
        curr_inds = Determinantal.sample(dpp, N′′)
        inds = unique([inds; curr_inds])
        curr_N′ = Base.size(inds, 1)
    end
    inds = inds[1:N′]

    return inds
end

# DPP-based sampling method
function dpp_sample(A, N′)
    # Compute a kernel matrix for the points in x
    #L = [ exp(-norm(a-b)^2) for a in eachcol(A'), b in eachcol(A') ]
    L = pairwise(Distances.Euclidean(), A')
    
    # Form an L-ensemble based on the L matrix
    dpp = EllEnsemble(L)
    
    # Scale so that the expected size is N′
    rescale!(dpp, N′)

    # Sample A (obtain indices)
    inds = Determinantal.sample(dpp)
     
    return inds
end

# CUR-based sampling method
function cur_sample(A, N′)
    r, _ = cur(A)
    inds = @views r
    if length(r) > N′
        inds = @views r[1:N′]
    end
    return inds
end

# Fit function used to get errors based on sampling
function fit(path, ds_train, ds_test, basis)

    # Learn
    lb = LBasisPotential(basis)
    ws, int = [1.0, 1.0], false
    learn!(lb, ds_train, ws, int)

    @save_var path lb.β
    @save_var path lb.β0

    # Post-process output: calculate metrics, create plots, and save results #######

    # Get true and predicted values
    n_atoms_train = length.(get_system.(ds_train))
    n_atoms_test = length.(get_system.(ds_test))

    e_train, e_train_pred = get_all_energies(ds_train) ./ n_atoms_train,
                            get_all_energies(ds_train, lb) ./ n_atoms_train
    f_train, f_train_pred = get_all_forces(ds_train),
                            get_all_forces(ds_train, lb)
    @save_var path e_train
    @save_var path e_train_pred
    @save_var path f_train
    @save_var path f_train_pred

    e_test, e_test_pred = get_all_energies(ds_test) ./ n_atoms_test,
                          get_all_energies(ds_test, lb) ./ n_atoms_test
    f_test, f_test_pred = get_all_forces(ds_test),
                          get_all_forces(ds_test, lb)
    @save_var path e_test
    @save_var path e_test_pred
    @save_var path f_test
    @save_var path f_test_pred

    # Compute metrics
    e_train_metrics = get_metrics(e_train, e_train_pred,
                                  metrics = [mae, rmse, rsq],
                                  label = "e_train")
    f_train_metrics = get_metrics(f_train, f_train_pred,
                                  metrics = [mae, rmse, rsq, mean_cos],
                                  label = "f_train")
    train_metrics = merge(e_train_metrics, f_train_metrics)
    @save_dict path train_metrics

    e_test_metrics = get_metrics(e_test, e_test_pred,
                                 metrics = [mae, rmse, rsq],
                                 label = "e_test")
    f_test_metrics = get_metrics(f_test, f_test_pred,
                                 metrics = [mae, rmse, rsq, mean_cos],
                                 label = "f_test")
    test_metrics = merge(e_test_metrics, f_test_metrics)
    @save_dict path test_metrics

    # Plot and save results

    e_plot = plot_energy(e_train, e_train_pred,
                         e_test, e_test_pred)
    @save_fig path e_plot

    f_plot = plot_forces(f_train, f_train_pred,
                         f_test, f_test_pred)
    @save_fig path f_plot

    e_train_plot = plot_energy(e_train, e_train_pred)
    f_train_plot = plot_forces(f_train, f_train_pred)
    f_train_cos  = plot_cos(f_train, f_train_pred)
    @save_fig path e_train_plot
    @save_fig path f_train_plot
    @save_fig path f_train_cos

    e_test_plot = plot_energy(e_test, e_test_pred)
    f_test_plot = plot_forces(f_test, f_test_pred)
    f_test_cos  = plot_cos(f_test, f_test_pred)
    @save_fig path e_test_plot
    @save_fig path f_test_plot
    @save_fig path f_test_cos
    
    return e_train_metrics, f_train_metrics, 
           e_test_metrics, f_test_metrics
end

# Main sample experiment function
function sample_experiment!(res_path, j, sampler, batch_size_prop, n_train, 
                            ged_mat, ds_train_rnd, ds_test_rnd, basis, metrics)
    try
        println("Experiment:$j, method:$sampler, batch_size_prop:$batch_size_prop")
        exp_path = "$res_path/$j-$sampler-bsp$batch_size_prop/"
        run(`mkdir -p $exp_path`)
        batch_size = floor(Int, n_train * batch_size_prop)
        sampling_time = @elapsed begin
            inds = sampler(ged_mat, batch_size)
        end
        metrics_j = fit(exp_path, (@views ds_train_rnd[Int64.(inds)]), ds_test_rnd, basis)
        metrics_j = merge(OrderedDict("exp_number" => j,
                                "method" => "$sampler",
                                "batch_size_prop" => batch_size_prop,
                                "batch_size" => batch_size,
                                "time" => sampling_time),
                    merge(metrics_j...))
        push!(metrics, metrics_j)
        @save_dataframe(res_path, metrics)
    catch e # Catch error from excessive matrix allocation.
        println(e)
    end
end

# Experiment j - DPP′ using n_chunks ##############################
# for n_chunks in [2, 4, 8]
#     println("Experiment:$j, method:DPP′(n=$n_chunks), batch_size_prop:$batch_size_prop")
#     exp_path = "$res_path/$j-DPP′-bsp$batch_size_prop-n$n_chunks/"
#     run(`mkdir -p $exp_path`)
#     inds = Int[]
#     n_chunk = n_train ÷ n_chunks
#     batch_size_chunk = floor(Int, n_chunk * batch_size_prop)
#     if batch_size_chunk == 0 
#         batch_size_chunk = 1
#     end
    
#     #sampling_time = @elapsed @threads for i in 1:n_threads
#     sampling_time = @elapsed for i in 1:n_chunks
#         a, b = 1 + (i-1) * n_chunk, i * n_chunk + 1
#         b = norm(b-n_train)<n_chunk ? n_train : b
#         inds_i = dpp_sample(@views(ged_mat[a:b, :]), batch_size_chunk)
#         append!(inds, inds_i .+ (a .- 1))
#     end
#     metrics_j = fit(exp_path, (@views ds_train_rnd[inds]), ds_test_rnd, basis)
#     metrics_j = merge(OrderedDict("exp_number" => j,
#                                   "method" => "DPP′(n:$n_chunks)",
#                                   "batch_size_prop" => batch_size_prop,
#                                   "batch_size" => batch_size,
#                                   "time" => sampling_time),
#                       merge(metrics_j...))
#     push!(metrics, metrics_j)
#     @save_dataframe(res_path, metrics)
# end