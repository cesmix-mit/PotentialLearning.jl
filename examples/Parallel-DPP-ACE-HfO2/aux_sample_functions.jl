
import Base.\
\(C::CholeskyPreconditioner, A::AbstractMatrix) = reduce(hcat, [C \ c for c in eachcol(A)])

function PotentialLearning.learn!(
    lp::PotentialLearning.CovariateLinearProblem,
    ws::Vector,
    int::Bool;
    λ::Real=0.0
)
    println("(using redefinition of learn)")
    @views B_train = reduce(hcat, lp.B)'
    @views dB_train = reduce(hcat, lp.dB)'
    @views e_train = lp.e
    @views f_train = reduce(vcat, lp.f)
    
    # Calculate A and b.
    if int
        #int_col = ones(size(B_train, 1) + size(dB_train, 1))
        int_col = [ones(Base.size(B_train, 1)); zeros(Base.size(dB_train, 1))]
        @views A = hcat(int_col, [B_train; dB_train])
    else
        @views A = [B_train; dB_train]
    end
    @views b = [e_train; f_train]

    # Calculate coefficients βs.
    Q = Diagonal([ws[1] * ones(length(e_train));
                  ws[2] * ones(length(f_train))])
    
    βs = Vector{Float64}() 
    try

        # Option 0:
        #A′ = (A'*Q*A + λ*I)
        #b′ = (A'*Q*b)
        #βs = A′ \ b′

        # Option 1:
        #A′ = (A'*Q*A + λ*I)
        #b′ = (A'*Q*b)
        #P = CholeskyPreconditioner(A′, 2)
        #P = DiagonalPreconditioner(A′)
        #P = AMGPreconditioner{SmoothedAggregation}(A)
        #βs = cg(A′, b′, Pl = P)

        # Option 2:
        #A′ = (A'*Q*A)
        #b′ = (A'*Q*b)
        #P = CholeskyPreconditioner(A′, 2)
        #P = DiagonalPreconditioner(A′)
        #P = AMGPreconditioner{SmoothedAggregation}(A)
        #βs = (A′ + λ*P'*P) \ b′
        
        # Option 3
        A′ = (A'*Q*A + λ*I)
        b′ = (A'*Q*b)
        P = CholeskyPreconditioner(A′, 2)
        CA = P \ A′
        Cb = P \ b′
        βs = CA \ Cb
    catch e
        println(e)
        println("Linear system will be solved using pinv.") 
        βs = pinv(A'*Q*A + λ*I)*(A'*Q*b)
    end

    # Update lp.
    if int
        lp.β0 .= βs[1]
        lp.β  .= βs[2:end]
    else
        lp.β  .= βs
    end
    
end


# Fit function used to get errors based on sampling
function fit(path, ds_train, ds_test, basis)

    # Learn
    lb = PotentialLearning.LBasisPotential(basis)
    ws, int = [30.0, 1.0], true
    #learn!(lb, ds_train, ws, int, λ=0.1)

    lp = PotentialLearning.LinearProblem(ds_train)
    learn!(lp, ws, int; λ=0.01)
    resize!(lb.β, length(lp.β))
    lb.β .= lp.β
    lb.β0 .= lp.β0
    
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
        print("Experiment:$j, method:$sampler, batch_size_prop:$batch_size_prop")
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
        print(", e_test_mae:$(round(metrics_j["e_test_mae"], sigdigits=4)), f_test_mae:$(round(metrics_j["f_test_mae"], sigdigits=4)), time:$(round(sampling_time, sigdigits=4))")
        println()
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