export train!, learn


"""
    learn(B_train, dB_train, e_train, f_train, w_e, w_f)

`B_train`: energy descriptors.
`dB_train`: force descriptors.
`e_train`: training energies.
`f_train`: training forces.
`w_e`: energy weight.
`w_f`: force weight.

"""
function learn(B_train, dB_train, e_train, f_train, w_e, w_f)

    # Calculate A and b.
    A = [B_train; dB_train]
    b = [e_train; f_train]


    # Calculate coefficients β.
    Q = Diagonal([w_e * ones(length(e_train));
                  w_f * ones(length(f_train))])
    β = (A'*Q*A) \ (A'*Q*b)


    ## Check weights.
    #using IterTools
    #for (e_weight, f_weight) in product(1:10:100, 1:10:100)
    #    Q = Diagonal([e_weight * ones(length(e_train));
    #                  f_weight * ones(length(f_train))])
    #    try
    #        β = (A'*Q*A) \ (A'*Q*b)
    #        a = compute_errors(dB_test * β, f_test)
    #        println(e_weight,", ", f_weight, ", ", a[1])
    #    catch
    #        println("Exception with :", e_weight,", ", f_weight)
    #    end
    #end

    return β
end


"""
    batch_train_opt!(ps, re, opt, maxiters, train_loader_e, train_loader_f,
                     w_e, w_f, epoch, train_losses_batches)

`ps`: neural network parameters. See Flux.destructure.
`re`: neural network restructure. See Flux.destructure.
`opt`: optimizer.
`maxiters`: maximum number of iterations in the optimizer.
`train_loader_e`: energy data loader.
`train_loader_f`: force data loader.
`w_e`: energy weight.
`w_f`: force weight.
`epoch`: current epoch.
`train_losses_batches`: vector of batch losses during training.

Batch training using Optimization.jl. Returns NN parameters.

"""
function batch_train_opt!(ps, re, opt, maxiters, train_loader_e, train_loader_f,
                          w_e, w_f, epoch, train_losses_batches)
    i = 1
    for ((bs_e, es), (bs_f, dbs_f, fs)) in zip(train_loader_e, train_loader_f)
        batch_loss(ps, p) = loss(potential_energy.(bs_e, [ps], [re]), es, w_e,
                                 force.(bs_f, dbs_f, [ps], [re]), fs, w_f)
        dbatchlossdps = OptimizationFunction(batch_loss,
                                             Optimization.AutoForwardDiff()) # Optimization.AutoZygote()
        prob = OptimizationProblem(dbatchlossdps, ps, []) # prob = remake(prob,u0=sol.minimizer)
        callback = function (p, l)
            println("Epoch: $(epoch), batch: $i, training loss: $l")
            push!(train_losses_batches, l)
            return false
        end
        sol = solve(prob, opt, callback=callback, maxiters=maxiters) # reltol = 1e-14
        ps = sol.u
        i = i + 1
    end
    return ps
end

"""
    batch_train_flux!(ps, re, opt, maxiters, train_loader_e, train_loader_f, w_e, w_f)

`ps`: neural network parameters. See Flux.destructure.
`re`: neural network restructure. See Flux.destructure.
`opt`: optimizer.
`maxiters`: maximum number of iterations in the optimizer.
`train_loader_e`: energy data loader.
`train_loader_f`: force data loader.
`w_e`: energy weight.
`w_f`: force weight.

Batch training using Flux.jl. Returns neural network parameters.

"""
function batch_train_flux!(ps, re, opt, maxiters, train_loader_e, train_loader_f, w_e, w_f)
    for ((bs_e, es), (bs_f, dbs_f, fs)) in zip(train_loader_e, train_loader_f)
        g = gradient(Flux.params(ps)) do
            loss(potential_energy.(bs_e, [ps], [re]), es, w_e,
                 force.(bs_f, dbs_f, [ps], [re]), fs, w_f)
        end
        Flux.Optimise.update!(opt, Flux.params(ps), g, maxiters=maxiters)
    end
    return ps
end

"""
    train!( lib, nnbp, epochs, opt, maxiters, train_loader_e, train_loader_f,
            test_loader_e, test_loader_f, w_e, w_f)

`lib`:
`nnbp`:  
`epochs`: no. of epochs.
`opt`: optimizer.
`maxiters`: maximum number of iterations in the optimizer.
`train_loader_e`: energy data loader for training.
`train_loader_f`: force data loader for training.
`test_loader_e`: energy data loader for test.
`test_loader_f`: force data loader for test.
`w_e`: energy weight.
`w_f`: force weight.

Train neural network potential.
Returns losses of training and test per epoch and per batch.

"""
function train!( lib, nnbp, epochs, opt, maxiters, train_loader_e, train_loader_f,
                 test_loader_e, test_loader_f, w_e, w_f)

    train_losses_epochs = []; test_losses_epochs = []; train_losses_batches = []
    ps, re = Flux.destructure(nnbp.nn)
    for epoch in 1:epochs
    
        # TODO: use multiple dispatch here?
        if lib == "Flux.jl"
            ps = batch_train_flux!(ps, re, opt, maxiters, train_loader_e,
                                   train_loader_f, w_e, w_f, epoch, train_losses_batches)
        else # lib == "Optimization.jl"
            ps = batch_train_opt!(ps, re, opt, maxiters, train_loader_e,
                                  train_loader_f, w_e, w_f, epoch, train_losses_batches)
        end
        
        # Report losses
        train_loss = global_loss(train_loader_e, train_loader_f, w_e, w_f, ps, re)
        test_loss = global_loss(test_loader_e, test_loader_f, w_e, w_f, ps, re)
        push!(train_losses_epochs, train_loss)
        push!(test_losses_epochs, test_loss)
        println("Epoch $(epoch). Losses of complete datasets: \
                 training loss: $(train_loss), \
                 test loss: $(test_loss).")
    end
    nnbp.nn = re(ps)
    nnbp.nn_params = Flux.params(nnbp.nn)
    
    return train_losses_epochs, test_losses_epochs, train_losses_batches
end


# Training using BFGS from Optimizer.jl (1 batch)
#time_fitting += Base.@elapsed begin
#    ps, re = Flux.destructure(nnbp.nn)
#    ((bs_e, es), (bs_f, dbs_f, fs)) = collect(zip(train_loader_e, train_loader_f))[1]
#    loss(ps, p) = loss(potential_energy.(bs_e, [ps], [re]), es, 
#                       force.(bs_f, dbs_f, [ps], [re]), fs)
#    dlossdps = OptimizationFunction(loss, Optimization.AutoForwardDiff()) # Optimization.AutoZygote()
#    prob = OptimizationProblem(dlossdps, ps, []) #prob = remake(prob,u0=sol.minimizer)
#    callback = function (p, l)
#        println("Thread: $(Threads.threadid()) current loss is: $l")
#        return false
#    end
#    sol = solve(prob, BFGS(), callback=callback, maxiters=1000) # reltol = 1e-14
#    ps = sol.u
#    nnbp.nn = re(ps)
#    nnbp.nn_params = Flux.params(nnbp.nn)
#end

#-------------------------------------------------------------------------------
## Training using BFGS and data parallelism with Base.Threads
# execute: julia --threads 4 fit-ahfo2-neural-ace.jl
# BLAS.set_num_threads(1)
##time_fitting += Base.@elapsed begin
#ps, re = Flux.destructure(nnbp.nn)
#nt = Threads.nthreads() 
#loaders = collect(zip(train_loader_e, train_loader_f))[1:nt]
## Compute loss of each batch
#opt_func = Array{Function}(undef, nt);
#for tid in 1:nt
#    ((bs_e, es), (bs_f, dbs_f, fs)) = loaders[tid]
#    batch_loss(ps, p) = loss(potential_energy.(bs_e, [ps], [re]), es, 
#                             force.(bs_f, dbs_f, [ps], [re]), fs)
#    opt_func[tid] = batch_loss
#end
## Compute total loss and define optimization function
#total_loss(ps, p) = mean([f(ps, p) for f in opt_func])
#dlossdps = OptimizationFunction(total_loss, Optimization.AutoForwardDiff()) # Optimization.AutoZygote())
## Optimize using averaged gradient on each batch
#callback = function (p, l)
#    println("Thread $(Threads.threadid()), current loss is: $l")
#    return false
#end
#pss = [deepcopy(ps) for i in 1:nt]
##@profile Threads.@threads for tid in 1:nt
#    ((bs_e, es), (bs_f, dbs_f, fs)) = loaders[tid]
#    ps_i = deepcopy(ps)
#    prob = OptimizationProblem(dlossdps, ps_i, []) # prob = remake(prob,u0=sol.minimizer)
#    sol = solve(prob, BFGS(), callback = callback, maxiters=10_000_000)
#    pss[tid] = sol.u
#end
## Average parameters
#ps = mean(pss)
#nnbp.nn = re(ps)
#nnbp.nn_params = Flux.params(nnbp.nn)
#end
#save("fit-ahfo2-neural-ace.jlprof",  Profile.retrieve()...)
#using ProfileView, FileIO
#data = load("fit-ahfo2-neural-ace.jlprof")
#ProfileView.view(data[1], lidict=data[2])
#-------------------------------------------------------------------------------

