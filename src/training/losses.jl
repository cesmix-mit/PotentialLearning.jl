export loss, global_loss


"""
    loss(x, y)
    
`x`: vector of scalars
`y`: vector of scalars

Returns the loss of two vectors

"""
loss(x, y) = Flux.Losses.mae(x, y)


"""
    loss(es_pred, es, w_e, fs_pred, fs, w_f)
    
`es_pred`: preditected energies
`es`: energies
`w_e`: energy weight
`fs_pred`: preditected forces
`fs`: forces
`w_f`: force weight

Returns the weighted loss of the energies and forces.

"""
loss(es_pred, es, w_e, fs_pred, fs, w_f) =  w_e * loss(es_pred, es) +
                                            w_f * loss(fs_pred, fs)


"""
    global_loss(loader_e, loader_f, w_e, w_f, ps, re)
    
`loader_e`: preditected energies
`loader_f`: energies
`w_e`: energy weight
`w_f`: force weight
`ps`: neural network parameters. See Flux.destructure.
`re`: neural network restructure. See Flux.destructure.

Returns the weighted global (all batches) loss of the energies and forces.

"""
global_loss(loader_e, loader_f, w_e, w_f, ps, re) =
    mean([loss(potential_energy.(bs_e, [ps], [re]), es, w_e,
               force.(bs_f, dbs_f, [ps], [re]), fs, w_f)
          for ((bs_e, es), (bs_f, dbs_f, fs)) in zip(loader_e, loader_f)])


"""
    global_energy_loss(loader_e, w_e, ps, re)
    
`loader_e`: preditected energies
`w_e`: energy weight
`ps`: neural network parameters. See Flux.destructure.
`re`: neural network restructure. See Flux.destructure.

Returns the weighted global (all batches) loss of the energies.

"""
global_energy_loss(loader_e, w_e, ps, re) =
    mean([ w_e * loss(potential_energy.(bs_e, [ps], [re]), es)
           for (bs_e, es) in loader_e])


"""
    global_force_loss(loader_f, w_f, ps, re)
    
`loader_f`: energies
`w_f`: force weight
`ps`: neural network parameters. See Flux.destructure.
`re`: neural network restructure. See Flux.destructure.

Returns the weighted global (all batches) loss of the forces.

"""
global_force_loss(loader_f, w_f, ps, re) =
    mean([ w_f * loss(force.(bs_f, dbs_f, [ps], [re]), fs)
           for (bs_f, dbs_f, fs) in loader_f])


