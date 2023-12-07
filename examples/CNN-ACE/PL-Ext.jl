# Aux. CNN functions ###########################################################

function PotentialLearning.get_all_energies(ds::DataSet, nnbp::NNBasisPotential)
    return nnbp.nns(get_e_descr_batch(ds))'
end

function get_e_descr_batch(ds)
    xs = []
    for c in ds
        ld_c = reduce(hcat, get_values(get_local_descriptors(c)))'
        #ld_c = ld_c[randperm(size(ld_c,1)),:]
        ld_c = cat( ld_c[:, 1:n_desc÷2], ld_c[:, n_desc÷2+1:end], dims=3 )
        if xs == []
            xs = ld_c
        else
            xs = cat(xs, ld_c, dims=4)
        end

#        ld_c = get_values(get_local_descriptors(c))
#        #ld_c = ld_c[randperm(length(ld_c))]
#        ld_c = cat( [Matrix(hcat(l[1:n_desc÷2], l[n_desc÷2+1:end])')
#                     for l in ld_c]..., dims=3)
#        
#        if xs == []
#            xs = ld_c
#        else
#            xs = cat(xs, ld_c, dims=4)
#        end

    end
    return xs
end

sqnorm(x) = sum(abs2, x)
function loss(x, y)
    return Flux.mse(x, y)
end

#function learn!(cnnnace, ds_train, opt, n_epochs, loss)
#    es = get_all_energies(ds_train) |> gpu
#    ld = get_e_descr_batch(ds_train) |> gpu
#    nn = cnnnace.nns |> gpu
#    opt = opt |> gpu
#    for epoch in 1:n_epochs
#        #grads = Flux.gradient(m -> loss(m(ld)', es) + sum(sqnorm, Flux.params(m)), nn)
#        grads = Flux.gradient(m -> loss(m(ld)', es), nn)
#        Flux.update!(opt, nn, grads[1])
#        if epoch % 100 == 0
#            #train_loss = loss(nn(ld)', es) + sum(sqnorm, Flux.params(nn))
#            train_loss = loss(nn(ld)', es)
#            println("epoch = $epoch; loss = $train_loss")
#        end
#    end
#    cnnnace.nns = nns |> cpu
#end

function PotentialLearning.learn!(cnnace, ds_train, ds_test, opt, n_epochs, loss)
    es = get_all_energies(ds_train) |> gpu
    ld = get_e_descr_batch(ds_train) |> gpu
    es_test = get_all_energies(ds_test) |> gpu
    ld_test = get_e_descr_batch(ds_test) |> gpu
    nns = cnnace.nns |> gpu
    opt = opt |> gpu
    for epoch in 1:n_epochs
        grads = Flux.gradient(m -> loss(m(ld)', es), nns)
        Flux.update!(opt, nns, grads[1])
        if epoch % 500 == 0
            train_loss = loss(nns(ld)', es)
            test_loss = loss(nns(ld_test)', es_test)
            println("epoch = $epoch; train loss = $train_loss, test loss = $test_loss")
        end
    end
    cnnace.nns = nns |> cpu
end

