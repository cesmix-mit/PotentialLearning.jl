using Flux
using Optim

# Neural network interatomic potential
mutable struct NNIAP
    nn
    iap
end


# Neural network potential formulation using global descriptors to compute energy
# See https://docs.google.com/presentation/d/1XI9zqF_nmSlHgDeFJqq2dxdVqiiWa4Z-WJKvj0TctEk/edit#slide=id.g169df3c161f_63_123

function potential_energy(c::Configuration, nnaip::NNIAP)
    Bs = sum(get_values(get_local_descriptors(c)))
    return sum(nnaip.nn(Bs))
end

function force(c::Configuration, nnaip::NNIAP)
    B = sum(get_values(get_local_descriptors(c)))
    dnndb = first(gradient(x->sum(nnaip.nn(x)), B)) 
    dbdr = get_values(get_force_descriptors(c))
    return [[-dnndb ⋅ dbdr[atom][coor] for coor in 1:3]
             for atom in 1:length(dbdr)]
end


# Neural network potential formulation using local descriptors to compute energy
# See 10.1103/PhysRevLett.98.146401
#     https://fitsnap.github.io/Pytorch.html

#function potential_energy(c::Configuration, nnaip::NNIAP)
#    Bs = get_values(get_local_descriptors(c))
#    return sum([sum(nnaip.nn(B)) for B in Bs])
#end

#function force(c::Configuration, nnaip::NNIAP)
#    Bs = get_values(get_local_descriptors(c))
#    dnndb = [first(gradient(x->sum(nnaip.nn(x)), B)) for B in Bs]
#    dbdr = get_values(get_force_descriptors(c))
#    return [[-dnndb[atom] ⋅ dbdr[atom][coor] for coor in 1:3]
#             for atom in 1:length(dbdr)]
#end


# Loss function ################################################################
function loss(nn, iap, ds, w_e = 1, w_f = 1)
    nniap = NNIAP(nn, iap)
    es, es_pred = get_all_energies(ds), get_all_energies(ds, nniap)
    fs, fs_pred = get_all_forces(ds), get_all_forces(ds, nniap)
    return w_e * Flux.mse(es_pred, es) + w_f * Flux.mse(fs_pred, fs)
end


# Auxiliary functions ##########################################################
function get_all_energies(ds::DataSet, nnaip::NNIAP)
    return [potential_energy(ds[c], nnaip) for c in 1:length(ds)]
end

function get_all_forces(ds::DataSet, nnaip::NNIAP)
    return reduce(vcat,reduce(vcat,[force(ds[c], nnaip)
                                    for c in 1:length(ds)]))
end

# NNIAP learning functions #####################################################

# Flux.jl training
function learn!(nnaip, ds, opt::Flux.Optimise.AbstractOptimiser, epochs, loss, w_e, w_f)
    optim = Flux.setup(opt, nnaip.nn)  # will store optimiser momentum, etc.
    ∇loss(nn, iap, ds, w_e, w_f) = gradient((nn) -> loss(nn, iap, ds, w_e, w_f), nn)
    losses = []
    for epoch in 1:epochs
        # Compute gradient with current parameters and update model
        grads = ∇loss(nnaip.nn, nnaip.iap, ds, w_e, w_f)
        Flux.update!(optim, nnaip.nn, grads[1])
        # Logging
        curr_loss = loss(nnaip.nn, nnaip.iap, ds, w_e, w_f)
        push!(losses, curr_loss)
        println(curr_loss)
    end
end

# Optimization.jl training
function learn!(nnaip, ds, opt::Optim.FirstOrderOptimizer, maxiters, loss, w_e, w_f)
    ps, re = Flux.destructure(nnaip.nn)
    batchloss(ps, p) = loss(re(ps), nnaip.iap, ds_train, w_e, w_f)
    opt = BFGS()
    ∇bacthloss = OptimizationFunction(batchloss, Optimization.AutoForwardDiff()) # Optimization.AutoZygote()
    prob = OptimizationProblem(∇bacthloss, ps, []) # prob = remake(prob,u0=sol.minimizer)
    cb = function (p, l) println("Loss BFGS: $l"); return false end
    sol = solve(prob, opt, maxiters=maxiters, callback = cb)
    ps = sol.u
    nn = re(ps)
    nnaip.nn = nn
    #copyto!(nnaip.nn, nn)
    #global nnaip = NNIAP(nn, nnaip.iap) # TODO: improve this
end



