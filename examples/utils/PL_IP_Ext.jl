using Flux
using JuLIP
using Zygote

# ToDo: compute forces for multielement approach and then integrate in InteratomicPotentials.jl

function PotentialLearning.force(
    c::Configuration,
    nnbp::NNBasisPotential
)
    s = get_system(c)
    a = InteratomicPotentials.convert_system_to_atoms(s)
    n_atoms = length(a)
    ns = JuLIP.neighbourlist(a, nnbp.basis.rcutoff)
    
    dNNdDi = Dict()
    for i = 1:n_atoms
        as = atomic_symbol(s)[i]
        #r = desc_range(i, s, basis)
        ledi = JuLIP.site_energy(basis.rpib, a, i)#[r]
        dNNdDi[i] = Zygote.gradient(x->sum(nnbp.nns[as](x)), ledi)[1]
    end

    fs = []
    for j in 1:n_atoms
        lfdj = JuLIP.site_energy_d(basis.rpib, a, j)
        nj, _ = neigs(ns, j)
        fs_j = []
        for α in 1:3
            f_j_α = 0.0
            for i in nj
                dDidrj_α = [lfdj[k][i][α] for k in 1:26]
                f_j_α += dNNdDi[i] ⋅ dDidrj_α
            end
            push!(fs_j, f_j_α)
        end
        push!(fs, fs_j)
    end
    
    return fs
end

#function force( # see PL/src/Data/utils.jl
#    c::Configuration,
#    nnbp::NNBasisPotential
#)
##    Bs = reduce(hcat, get_values(get_local_descriptors(c))) # can be precomputed
##    dnndb = first(gradient(x->sum(nnbp.nn(x)), Bs))        # gradient function of MLP can be predefined
##    dbdr = reduce(hcat, get_values(get_force_descriptors(c))) # can be precomputed
##    n_atoms = size(dbdr, 2)
##    return [[-sum( dot.(eachcol(dnndb), [dbdr[coor, atom_j]]) )
##             for coor in 1:3]
##             for atom_j in 1:n_atoms]
#    
##    Bs = reduce(hcat, get_values(get_local_descriptors(c))) # can be precomputed
##    dnndb = first(gradient(x->sum(nnbp.nn(x)), Bs))        # gradient function of MLP can be predefined
##    sum_dnndb = sum(dnndb, dims = 2)
##    dbdr = reduce(hcat, get_values(get_force_descriptors(c))) # can be precomputed
##    n_atoms = size(dbdr, 2)
##    return [[-(sum_dnndb ⋅ dbdr[coor, atom_j] / n_atoms) for coor in 1:3]
##             for atom_j in 1:n_atoms]
#    
#    Bs = reduce(hcat, get_values(get_local_descriptors(c))) # can be precomputed
#    dnndb = first(Flux.gradient(x->sum(nnbp.nn(x)), Bs))   # gradient function of MLP can be predefined
#    n_atoms = size(dnndb, 2)
#    global dbdr_c
#    return [[ -sum(dot.(eachcol(dnndb), eachcol(dbdr_c[c][coor, atom_j])))
#              for coor in 1:3]
#              for atom_j in 1:n_atoms]

##    Bs = get_values(get_local_descriptors(c))
##    dnndb = [first(gradient(x->sum(nnbp.nn(x)), B_atom)) for B_atom in Bs]
##    dbdr = get_values(get_force_descriptors(c))
##    return [[-sum(dnndb .⋅ [dbdr[atom][coor]]) for coor in 1:3] for atom in 1:length(dbdr)]
#end


#function force(
#    c::Configuration,
#    nnbp::NNBasisPotential
#)
#    Bs = reduce(hcat, get_values(get_local_descriptors(c))) # can be precomputed
#    dnnsdb = [first(Flux.gradient(x->sum(nn(x)), Bs)) for nn in nnbp.nns] # gradient function of MLP can be predefined

#    n_atoms = size(dnndb, 2)
#    global dbdr_c
#    return [[ -sum(dot.(eachcol(dnndb), eachcol(dbdr_c[c][coor, atom_j])))
#              for coor in 1:3]
#              for atom_j in 1:n_atoms]
#end


# ToDo: Integrate the code below in PotentialLearning.jl

# Loss functions ###############################################################

function energy_loss(
    nns::Dict,
    basis::BasisSystem,
    ds::DataSet,
    args...
)
    nnbp = NNBasisPotential(nns, basis)
    n_atoms = [length(get_local_descriptors(ds[i])) for i in 1:length(ds)]
    es, es_pred = get_all_energies(ds) ./ n_atoms,
                  get_all_energies(ds, nnbp) ./ n_atoms
    return Flux.mse(es_pred, es)
end

function loss(
    nns::Dict,
    basis::BasisSystem,
    ds::DataSet,
    w_e::Real = 1.0,
    w_f::Real = 1.0
)
    nnbp = NNBasisPotential(nns, basis)
    es, es_pred = get_all_energies(ds), get_all_energies(ds, nnbp)  # get_all_energies(ds) can be precomputed
    fs, fs_pred = get_all_forces(ds), get_all_forces(ds, nnbp)      # get_all_forces(ds) can be precomputed
    return w_e * Flux.mse(es_pred, es) + w_f * Flux.mse(fs_pred, fs)
end

# Auxiliary functions ##########################################################

function PotentialLearning.get_system(c::Configuration)
    for k in keys(c.data)
        if k <: AtomsBase.AbstractSystem
            return c.data[k]
        end
    end
end

# nnbp learning functions #####################################################

# Flux.jl training
function PotentialLearning.learn!(
    nnbp::NNBasisPotential,
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
    optim = Flux.setup(OptimiserChain(WeightDecay(reg), opt), nnbp.nns)
    ∇loss(nns, basis, ds, w_e, w_f) = Flux.gradient((nns) -> loss0(nns, basis, ds, w_e, w_f), nns)
    losses = []
    n_batches = length(ds) ÷ batch_size
    for epoch in 1:epochs
        for _ in 1:n_batches
            # Compute gradient with current parameters and update model
            batch_inds = rand(1:length(ds), batch_size)
            grads = ∇loss(nnbp.nns, nnbp.basis, ds[batch_inds], w_e, w_f)
            Flux.update!(optim, nnbp.nns, grads[1])
        end
        # Logging
        if epoch % log_step == 0
            curr_loss = loss0(nnbp.nns, nnbp.basis, ds, w_e, w_f)
            push!(losses, curr_loss)
            println("Epoch: $epoch, loss: $curr_loss")
        end
        GC.gc()
    end
end

# Compatibility with Optimization.jl. ##########################################
# TODO: update to multielement training.
#function PotentialLearning.learn!(
#    nnbp::nnbp,
#    ds::DataSet,
#    opt::Optim.FirstOrderOptimizer,
#    maxiters::Int,
#    loss::Function,
#    w_e::Real,
#    w_f::Real
#)
#    ps, re = Flux.destructure(nnbp.nn)
#    batchloss(ps, p) = loss(re(ps), nnbp.iap, ds, w_e, w_f)
#    ∇bacthloss = OptimizationFunction(batchloss, Optimization.AutoForwardDiff()) # Optimization.AutoZygote()
#    prob = OptimizationProblem(∇bacthloss, ps, []) # prob = remake(prob,u0=sol.minimizer)
#    cb = function (p, l) println("Loss BFGS: $l"); GC.gc(); return false end
#    sol = solve(prob, opt, maxiters=maxiters, callback = cb)
#    ps = sol.u
#    nn = re(ps)
#    nnbp.nn = nn
#    #copyto!(nnbp.nn, nn)
#    #global nnbp = nnbp(nn, nnbp.iap) # TODO: improve this
#end

# GPU training #################################################################

#function batch_and_shuffle(
#    data::DataSet,
#    num_batches::Int
#)
#    # Shuffle the data
#    shuffle!(data)

#    # Calculate the number of batches
#    batch_size = ceil(Int, length(data) / num_batches)
#    # Create the batches
#    batches = [data[(i-1)*batch_size+1:min(i*batch_size, end)] for i in 1:num_batches]

#    return batches
#end

#function loss(
#    nn::Chain,
#    iap::BasisSystem,
#    atom_config_list::Vector,
#    true_energy::Vector,
#    local_descriptors::Vector,
#    w_e::Real = 1.0,
#    w_f::Real = 1.0
#)
#    nn_local_descriptors = nn(local_descriptors)
#    atom_descriptors_list = [nn_local_descriptors[:, atom_config_list[i]+1:atom_config_list[i+1]] for i in 1:length(atom_config_list)-1]
#    atom_sum_pred = sum.(atom_descriptors_list)
#    return w_e * Flux.mse(atom_sum_pred, true_energy)
#end

# TODO: update to multielement training
#function PotentialLearning.learn!(
#    nace::nnbp,
#    ds_train::DataSet,
#    opt::Flux.Optimise.AbstractOptimiser,
#    n_epochs::Int,
#    n_batches::Int,
#    loss::Function,
#    w_e::Real,
#    w_f::Real,
#    _device::Function
#)
#    nn = nace.nn |> _device
#    optim = Flux.setup(opt, nn) |> _device
#    ∇loss(nn, iap, atom_config_list, true_energy, local_descriptors, w_e, w_f) = gradient((nn) -> loss(nn, iap, atom_config_list,  true_energy, local_descriptors, w_e, w_f), nn)
#    losses = []
#    batch_lists = batch_and_shuffle(collect(1:length(ds_train)), n_batches)
#    batch_list_len = length(batch_lists)
#    
#    for epoch in 1:n_epochs
#        batch_index = mod(epoch, batch_list_len) + 1 
#        ds_batch = ds_train[batch_lists[batch_index]]

#        true_energy = Float32.(get_all_energies(ds_batch))
#        
#        local_descriptors = get_values.(get_local_descriptors.(ds_batch))
#        local_descriptors = reduce(hcat, reduce(hcat, local_descriptors)) |> _device

#        atom_config_list = vcat([0], cumsum(length.(get_system.(ds_batch))))

#        # Compute gradient with current parameters and update model
#        grads = ∇loss(nn, nace.iap, atom_config_list, true_energy, local_descriptors, w_e, w_f)
#        Flux.update!(optim, nn, grads[1])

#        # Logging
#        if epoch % 100 == 0
#            curr_loss = loss(nn, nace.iap, atom_config_list, true_energy, local_descriptors, w_e, w_f)
#            push!(losses, curr_loss)
#            println("Epoch: $epoch, loss: $curr_loss")
#        end
#        
#    end
#    nace.nn = nn |> cpu
#end

