# Loss function
loss(es_pred, es, w_e, fs_pred, fs, w_f) =  w_e * Flux.Losses.mse(es_pred, es) +
                                            w_f * Flux.Losses.mse(fs_pred, fs)

# Global loss function
global_loss(loader_e, loader_f, w_e, w_f, ps, re) =
    mean([loss(potential_energy.(bs_e, [ps], [re]), es, w_e,
               force.(bs_f, dbs_f, [ps], [re]), fs, w_f)
          for ((bs_e, es), (bs_f, dbs_f, fs)) in zip(loader_e, loader_f)])

