export train!

"""
    batch_train_opt!(ps, re, opt, maxiters, train_loader_e, train_loader_f,
                     w_e, w_f, epoch, train_losses_batches)

TODO: complete documentation
Batch training using Optimization.jl
I am using one of the Optimization.jl solvers, because I have not been able to
tackle this problem with the Flux solvers.

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
    
TODO: complete documentation
Batch training using Flux.jl

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

TODO: complete documentation

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

