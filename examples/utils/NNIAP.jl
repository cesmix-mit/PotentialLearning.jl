using Flux
using Optim, Optimization

# Neural network interatomic potential #########################################

mutable struct NNIAP <: AbstractPotential
    nn
    iap
end

# Formulation using local descriptors to compute energy and forces
# See 10.1103/PhysRevLett.98.146401, https://fitsnap.github.io/Pytorch.html

function potential_energy(
    c::Configuration,
    nniap::NNIAP
)
#    Bs = get_values(get_local_descriptors(c))
#    s = sum(sum([nniap.nn(B_atom) for B_atom in Bs]))
#    return s
    Bs = reduce(hcat, get_values(get_local_descriptors(c))) # can be precomputed
    return sum(nniap.nn(Bs))
end

function force(
    c::Configuration,
    nniap::NNIAP
)
#    Bs = reduce(hcat, get_values(get_local_descriptors(c))) # can be precomputed
#    dnndb = first(gradient(x->sum(nniap.nn(x)), Bs))        # gradient function of MLP can be predefined
#    dbdr = reduce(hcat, get_values(get_force_descriptors(c))) # can be precomputed
#    n_atoms = size(dbdr, 2)
#    return [[-sum( dot.(eachcol(dnndb), [dbdr[coor, atom_j]]) )
#             for coor in 1:3]
#             for atom_j in 1:n_atoms]
    
#    Bs = reduce(hcat, get_values(get_local_descriptors(c))) # can be precomputed
#    dnndb = first(gradient(x->sum(nniap.nn(x)), Bs))        # gradient function of MLP can be predefined
#    sum_dnndb = sum(dnndb, dims = 2)
#    dbdr = reduce(hcat, get_values(get_force_descriptors(c))) # can be precomputed
#    n_atoms = size(dbdr, 2)
#    return [[-(sum_dnndb ⋅ dbdr[coor, atom_j] / n_atoms) for coor in 1:3]
#             for atom_j in 1:n_atoms]
    
    Bs = reduce(hcat, get_values(get_local_descriptors(c))) # can be precomputed
    dnndb = first(Flux.gradient(x->sum(nniap.nn(x)), Bs))   # gradient function of MLP can be predefined
    n_atoms = size(dnndb, 2)
    global dbdr_c
    return [[ -sum(dot.(eachcol(dnndb), eachcol(dbdr_c[c][coor, atom_j])))
              for coor in 1:3]
              for atom_j in 1:n_atoms]

#    Bs = get_values(get_local_descriptors(c))
#    dnndb = [first(gradient(x->sum(nniap.nn(x)), B_atom)) for B_atom in Bs]
#    dbdr = get_values(get_force_descriptors(c))
#    return [[-sum(dnndb .⋅ [dbdr[atom][coor]]) for coor in 1:3] for atom in 1:length(dbdr)]
end

# Formulation using global descriptors to compute energy and forces
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

function loss(
    nn::Chain,
    iap::BasisSystem,
    ds::DataSet,
    w_e::Real = 1.0,
    w_f::Real = 1.0
)
    nniap = NNIAP(nn, iap)
    es, es_pred = get_all_energies(ds), get_all_energies(ds, nniap)  # get_all_energies(ds) can be precomputed
    fs, fs_pred = get_all_forces(ds), get_all_forces(ds, nniap)      # get_all_forces(ds) can be precomputed
    return w_e * Flux.mse(es_pred, es) + w_f * Flux.mse(fs_pred, fs)
end

function loss(
    nn::Chain,
    iap::BasisSystem,
    atom_config_list::Vector,
    true_energy::Vector,
    local_descriptors::Vector,
    w_e::Real = 1.0,
    w_f::Real = 1.0
)
    nn_local_descriptors = nn(local_descriptors)
    atom_descriptors_list = [nn_local_descriptors[:, atom_config_list[i]+1:atom_config_list[i+1]] for i in 1:length(atom_config_list)-1]
    atom_sum_pred = sum.(atom_descriptors_list)
    return w_e * Flux.mse(atom_sum_pred, true_energy)
end


# Auxiliary functions ##########################################################

function PotentialLearning.get_all_energies(
    ds::DataSet,
    nniap::NNIAP
)
    return [potential_energy(ds[i], nniap) for i in 1:length(ds)]
end

function PotentialLearning.get_all_forces(
    ds::DataSet,
    nniap::NNIAP
)
    return reduce(vcat,reduce(vcat,[force(ds[c], nniap) for c in 1:length(ds)]))
end

function batch_and_shuffle(
    data::DataSet,
    num_batches::Int
)
    # Shuffle the data
    shuffle!(data)

    # Calculate the number of batches
    batch_size = ceil(Int, length(data) / num_batches)
    # Create the batches
    batches = [data[(i-1)*batch_size+1:min(i*batch_size, end)] for i in 1:num_batches]

    return batches
end

# NNIAP learning functions #####################################################

# Flux.jl training
function PotentialLearning.learn!(
    nniap::NNIAP,
    ds::DataSet,
    opt::Flux.Optimise.AbstractOptimiser,
    epochs::Int,
    loss0::Function,
    w_e::Real,
    w_f::Real,
    log_step::Int
)
    optim = Flux.setup(opt, nniap.nn)  # will store optimiser momentum, etc.
    ∇loss(nn, iap, ds, w_e, w_f) = gradient((nn) -> loss0(nn, iap, ds, w_e, w_f), nn)
    losses = []
    for epoch in 1:epochs
        # Compute gradient with current parameters and update model
        grads = ∇loss(nniap.nn, nniap.iap, ds, w_e, w_f)
        Flux.update!(optim, nniap.nn, grads[1])
        # Logging
        if epoch % log_step == 0
            curr_loss = loss0(nniap.nn, nniap.iap, ds, w_e, w_f)
            push!(losses, curr_loss)
            println("Epoch: $epoch, loss: $curr_loss")
            GC.gc()
        end
    end
end

function PotentialLearning.learn!(
    nniap::NNIAP,
    ds::DataSet,
    opt::Flux.Optimise.AbstractOptimiser,
    epochs::Int,
    loss0::Function,
    w_e::Real,
    w_f::Real,
    reg::Real,
    batch_size::Int,
    log_step::Int
)
    #optim = Flux.setup(opt, nniap.nn)  # will store optimiser momentum, etc.
    optim = Flux.setup(OptimiserChain(WeightDecay(reg), opt), nniap.nn)
    ∇loss(nn, iap, ds, w_e, w_f) = Flux.gradient((nn) -> loss0(nn, iap, ds, w_e, w_f), nn)
    losses = []
    n_batches = length(ds) ÷ batch_size
    for epoch in 1:epochs
        for _ in 1:n_batches
            # Compute gradient with current parameters and update model
            batch_inds = rand(1:length(ds), batch_size)
            grads = ∇loss(nniap.nn, nniap.iap, ds[batch_inds], w_e, w_f)
            Flux.update!(optim, nniap.nn, grads[1])
        end
        # Logging
        if epoch % log_step == 0
            curr_loss = loss0(nniap.nn, nniap.iap, ds, w_e, w_f)
            push!(losses, curr_loss)
            println("Epoch: $epoch, loss: $curr_loss")
        end
        GC.gc()
    end
end

function PotentialLearning.learn!(
    nace::NNIAP,
    ds_train::DataSet,
    opt::Flux.Optimise.AbstractOptimiser,
    n_epochs::Int,
    n_batches::Int,
    loss::Function,
    w_e::Real,
    w_f::Real,
    _device::Function
)
    nn = nace.nn |> _device
    optim = Flux.setup(opt, nn) |> _device
    ∇loss(nn, iap, atom_config_list, true_energy, local_descriptors, w_e, w_f) = gradient((nn) -> loss(nn, iap, atom_config_list,  true_energy, local_descriptors, w_e, w_f), nn)
    losses = []
    batch_lists = batch_and_shuffle(collect(1:length(ds_train)), n_batches)
    batch_list_len = length(batch_lists)
    
    for epoch in 1:n_epochs
        batch_index = mod(epoch, batch_list_len) + 1 
        ds_batch = ds_train[batch_lists[batch_index]]

        true_energy = Float32.(get_all_energies(ds_batch))
        
        local_descriptors = get_values.(get_local_descriptors.(ds_batch))
        local_descriptors = reduce(hcat, reduce(hcat, local_descriptors)) |> _device

        atom_config_list = vcat([0], cumsum(length.(get_system.(ds_batch))))

        # Compute gradient with current parameters and update model
        grads = ∇loss(nn, nace.iap, atom_config_list, true_energy, local_descriptors, w_e, w_f)
        Flux.update!(optim, nn, grads[1])

        # Logging
        if epoch % 100 == 0
            curr_loss = loss(nn, nace.iap, atom_config_list, true_energy, local_descriptors, w_e, w_f)
            push!(losses, curr_loss)
            println("Epoch: $epoch, loss: $curr_loss")
        end
        
    end
    nace.nn = nn |> cpu
end

# Optimization.jl training
function PotentialLearning.learn!(
    nniap::NNIAP,
    ds::DataSet,
    opt::Optim.FirstOrderOptimizer,
    maxiters::Int,
    loss::Function,
    w_e::Real,
    w_f::Real
)
    ps, re = Flux.destructure(nniap.nn)
    batchloss(ps, p) = loss(re(ps), nniap.iap, ds, w_e, w_f)
    ∇bacthloss = OptimizationFunction(batchloss, Optimization.AutoForwardDiff()) # Optimization.AutoZygote()
    prob = OptimizationProblem(∇bacthloss, ps, []) # prob = remake(prob,u0=sol.minimizer)
    cb = function (p, l) println("Loss BFGS: $l"); GC.gc(); return false end
    sol = solve(prob, opt, maxiters=maxiters, callback = cb)
    ps = sol.u
    nn = re(ps)
    nniap.nn = nn
    #copyto!(nniap.nn, nn)
    #global nniap = NNIAP(nn, nniap.iap) # TODO: improve this
end
