export loss, global_loss

"""
    loss(es_pred, es, w_e, fs_pred, fs, w_f)
    
`es_pred`: preditected energies
`es`: energies
`w_e`: energy weight
`fs_pred`: preditected forces
`fs`: forces
`w_f`: force weight

Returns the weighted loss of the energies and forces using MSE.

"""
loss(es_pred, es, w_e, fs_pred, fs, w_f) =  w_e * Flux.Losses.mse(es_pred, es) +
                                            w_f * Flux.Losses.mse(fs_pred, fs)

"""
    global_loss(loader_e, loader_f, w_e, w_f, ps, re)
    
`loader_e`: preditected energies
`loader_f`: energies
`w_e`: energy weight
`w_f`: force weight
`ps`: neural network parameters. See Flux.destructure.
`re`: neural network restructure. See Flux.destructure.

Returns the weighted global (all bacthes) loss of the energies and forces.

"""
global_loss(loader_e, loader_f, w_e, w_f, ps, re) =
    mean([loss(potential_energy.(bs_e, [ps], [re]), es, w_e,
               force.(bs_f, dbs_f, [ps], [re]), fs, w_f)
          for ((bs_e, es), (bs_f, dbs_f, fs)) in zip(loader_e, loader_f)])

