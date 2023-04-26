using Flux
using Optim
using Zygote


# Neural network interatomic potential
mutable struct NNIAP
    nn
    iap
end

# Neural network potential formulation using local descriptors to compute energy and forces
# See 10.1103/PhysRevLett.98.146401
#     https://fitsnap.github.io/Pytorch.html

function potential_energy(c::Configuration, nniap::NNIAP, _device=gpu)
    Bs = get_values(get_local_descriptors(c)) |> _device
    s = sum(sum([nniap.nn(B_atom) for B_atom in Bs]))
    return s
end

function potential_energy(c::Configuration, nniap::NNIAP)
    Bs = get_values(get_local_descriptors(c))
    s = sum([sum(nniap.nn(B_atom)) for B_atom in Bs])
    return s
end

function force(c::Configuration, nniap::NNIAP, _device)
    if _device != gpu
        return force(c, nniap)
    end

    Bs = get_values(get_local_descriptors(c))
    nniap.nn = nniap.nn |> cpu
    dnndb = [first(gradient(x->sum(nniap.nn(x)), B_atom)) for B_atom in Bs]
    dbdr = get_values(get_force_descriptors(c)) 
    return [[-sum(dnndb .⋅ [dbdr[atom][coor]]) for coor in 1:3] for atom in 1:length(dbdr)] |> gpu
end

function force(c::Configuration, nniap::NNIAP)
    Bs = get_values(get_local_descriptors(c))
    dnndb = [first(gradient(x->sum(nniap.nn(x)), B_atom)) for B_atom in Bs]
    dbdr = get_values(get_force_descriptors(c))
    return [[-sum(dnndb .⋅ [dbdr[atom][coor]]) for coor in 1:3] for atom in 1:length(dbdr)]
end

# Neural network potential formulation using global descriptors to compute energy and forces
# See https://docs.google.com/presentation/d/1XI9zqF_nmSlHgDeFJqq2dxdVqiiWa4Z-WJKvj0TctEk/edit#slide=id.g169df3c161f_63_123

#function potential_energy(c::Configuration, nniap::NNIAP)
#    Bs = sum(get_values(get_local_descriptors(c)))
#    return sum(nniap.nn(Bs))
#end

#function force(c::Configuration, nniap::NNIAP)
#    B = sum(get_values(get_local_descriptors(c)))
#    dnndb = first(gradient(x->sum(nniap.nn(x)), B)) 
#    dbdr = get_values(get_force_descriptors(c))
#    return [[-dnndb ⋅ dbdr[atom][coor] for coor in 1:3]
#             for atom in 1:length(dbdr)]
#end


# Loss function ################################################################
function gpu_loss(nn, iap, ds, w_e = 1, w_f = 1)
    _device = gpu
    nniap = NNIAP(nn, iap)
    println(ds)
    es, es_pred = get_all_energies(ds), get_all_energies(ds, nniap, _device)
    fs, fs_pred = get_all_forces(ds), get_all_forces(ds, nniap, _device)
    #println(es_pred)
    #es = es |> cpu
    # es_pred = es_pred |> cpu

    e_error = w_e * Flux.mse(es_pred, es) |> cpu
    fs = fs|> gpu
    f_error = w_f * Flux.mse(fs_pred, fs) |> cpu
    total_error = e_error + f_error
    return total_error
end
 

function loss(nn, iap, ds, w_e = 1, w_f = 1)
    nniap = NNIAP(nn, iap)
    es, es_pred = get_all_energies(ds), get_all_energies(ds, nniap)
    fs, fs_pred = get_all_forces(ds), get_all_forces(ds, nniap)
    return w_e * Flux.mse(es_pred, es) + w_f * Flux.mse(fs_pred, fs)
end


# Auxiliary functions ##########################################################
function get_all_energies(ds::DataSet, nniap::NNIAP, _device)
    return [potential_energy(ds[c], nniap, _device) for c in 1:length(ds)]
end


function get_all_energies(ds::DataSet, nniap::NNIAP)
    return [potential_energy(ds[c], nniap) for c in 1:length(ds)]
end

function get_all_forces(ds::DataSet, nniap::NNIAP, _device)
    return reduce(vcat,reduce(vcat,[force(ds[c], nniap, _device) for c in 1:length(ds)]))
end

function get_all_forces(ds::DataSet, nniap::NNIAP)
    return reduce(vcat,reduce(vcat,[force(ds[c], nniap) for c in 1:length(ds)]))
end

# NNIAP learning functions #####################################################

# Flux.jl training
function learn!(nniap, ds, opt::Flux.Optimise.AbstractOptimiser, epochs, loss, w_e, w_f)
    optim = Flux.setup(opt, nniap.nn)  # will store optimiser momentum, etc.
    ∇loss(nn, iap, ds, w_e, w_f) = gradient((nn) -> loss(nn, iap, ds, w_e, w_f), nn)
    losses = []
    for epoch in 1:epochs
        # Compute gradient with current parameters and update model
        grads = ∇loss(nniap.nn, nniap.iap, ds, w_e, w_f)
        Flux.update!(optim, nniap.nn, grads[1])
        # Logging
        curr_loss = loss(nniap.nn, nniap.iap, ds, w_e, w_f)
        push!(losses, curr_loss)
        println(curr_loss)
    end
end

function learn!(nniap, ds, opt::Flux.Optimise.AbstractOptimiser, epochs, loss, w_e, w_f,_device)
    if _device != gpu
        println("nooooooo")
        return learn!(nniap, ds, opt, epochs, loss, w_e, w_f)
    end

    optim = Flux.setup(opt, nniap.nn)
    nniap.nn = nniap.nn|> gpu
    nniap.iap = nniap.iap|> gpu
    ds  = ds |> gpu
    optim = Flux.gpu(optim)

    ∇gpu_loss(nn, iap, ds, w_e, w_f) = gradient((nn) -> gpu_loss(nn, iap, ds, w_e, w_f), nn)
    losses = []
    for epoch in 1:epochs
        # Compute gradient with current parameters and update model
        grads = ∇gpu_loss(nniap.nn, nniap.iap, ds, w_e, w_f)
        Flux.update!(optim, nniap.nn, grads[1])
        # Logging
        curr_loss = gpu_loss(nniap.nn, nniap.iap, ds, w_e, w_f)
        push!(losses, curr_loss)
        println(curr_loss)
    end
end

# Optimization.jl training
function learn!(nniap, ds, opt::Optim.FirstOrderOptimizer, maxiters, loss, w_e, w_f)
    ps, re = Flux.destructure(nniap.nn)
    batchloss(ps, p) = loss(re(ps), nniap.iap, ds, w_e, w_f)
    ∇bacthloss = OptimizationFunction(batchloss, Optimization.AutoForwardDiff()) # Optimization.AutoZygote()
    prob = OptimizationProblem(∇bacthloss, ps, []) # prob = remake(prob,u0=sol.minimizer)
    cb = function (p, l) println("Loss BFGS: $l"); return false end
    sol = solve(prob, opt, maxiters=maxiters, callback = cb)
    ps = sol.u
    nn = re(ps)
    nniap.nn = nn
    #copyto!(nniap.nn, nn)
    #global nniap = NNIAP(nn, nniap.iap) # TODO: improve this
end



