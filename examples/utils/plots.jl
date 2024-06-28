# TODO: units are hardcoded

"""
    plot_energy(
        e_true,
        e_pred
    )

`e_true`: vector of true energies
`e_pred`: vector of predicted energies

Returns an true vs predicted energy plot.

"""
function plot_energy(
    e_true,
    e_pred
)
    es =[e_true; e_pred]
    r0 = minimum(es)
    r1 = maximum(es)
    rs = (r1-r0)/10
    plot(e_true,
         e_pred, 
         seriestype = :scatter,
         dpi = 300,
         alpha = 0.35,
         thickness_scaling = 1.35, 
         markersize = 3,
         markerstrokewidth = 1,
         markerstrokecolor = :black,
         markershape = :circle,
         markercolor = :gray,
         label = "",
         xlabel = "E DFT | eV/atom",
         ylabel = "E predicted | eV/atom")
    p = plot!(r0:rs:r1,
              r0:rs:r1,
              label = "",
              color = :red)
    return p
end


"""
    plot_energy(
        e_train_true,
        e_train_pred
        e_test_true,
        e_test_pred
    )

`e_train_true`: vector of true training energies
`e_train_pred`: vector of predicted training energies
`e_test_true`: vector of true test energies
`e_test_pred`: vector of predicted test energies

Returns an true vs predicted energy plot for training and test.

"""
function plot_energy(
    e_train_true,
    e_train_pred,
    e_test_true,
    e_test_pred
)
    es =[e_train_true; e_train_pred; e_test_true; e_test_pred]
    r0 = minimum(es)
    r1 = maximum(es)
    rs = (r1-r0)/10
    plot(e_train_true,
         e_train_pred,
         seriestype = :scatter,
         alpha = 0.35,
         thickness_scaling = 1.35,
         markersize = 3,
         markerstrokewidth = 1,
         markerstrokecolor = :black,
         markershape = :circle,
         markercolor = :gray,
         label="Training energies")
    plot!(e_test_true,
          e_test_pred,
          seriestype = :scatter,
          alpha = 0.35,
          thickness_scaling = 1.35,
          markersize = 3,
          markerstrokewidth = 1,
          markerstrokecolor = :red,
          markershape = :utriangle,
          markercolor = :red2,
          label = "Test energies")
    plot!(dpi = 300,
          label = "",
          xlabel = "E DFT | eV/atom",
          ylabel = "E predicted | eV/atom")
    p = plot!(r0:rs:r1,
              r0:rs:r1,
              label = "", 
              color = :red)
    return p
end


"""
    plot_forces(
        f_true,
        f_pred
    )

`f_true`: vector of true forces
`f_pred`: vector of predicted forces

Returns an true vs predicted force plot.

"""
function plot_forces(
    f_true,
    f_pred
)
    r0 = floor(minimum(f_true))
    r1 = ceil(maximum(f_true))
    plot(f_true,
         f_pred, 
         seriestype = :scatter,
         alpha = 0.35,
         dpi = 300,
         thickness_scaling = 1.35,
         markersize = 3,
         markerstrokewidth = 1,
         markerstrokecolor = :black,
         markershape = :circle,
         markercolor = :gray, 
         label = "",
         xlabel = "F DFT | eV/Å",
         ylabel = "F predicted | eV/Å",
         xlims = (r0, r1),
         ylims = (r0, r1))
    p = plot!(r0:r1,
              r0:r1,
              label = "",
              color = :red)
    return p
end


"""
    plot_forces(
        f_train_true,
        f_train_pred
        f_test_true,
        f_test_pred
    )

`f_train_true`: vector of true training forces
`f_train_pred`: vector of predicted training forces
`f_test_true`: vector of true test forces
`f_test_pred`: vector of predicted test forces

Returns an true vs predicted force plot for training and test.

"""
function plot_forces(
    f_train_true,
    f_train_pred,
    f_test_true,
    f_test_pred
)
    fs =[f_train_true; f_train_pred; f_test_true; f_test_pred]
    r0 = minimum(fs)
    r1 = maximum(fs)
    rs = (r1-r0)/10
    plot(f_train_true,
         f_train_pred,
         seriestype = :scatter,
         alpha = 0.35,
         thickness_scaling = 1.35,
         markersize = 3,
         markerstrokewidth = 1,
         markerstrokecolor = :black,
         markershape = :circle,
         markercolor = :gray,
         label="Training forces")
    plot!(f_test_true,
          f_test_pred,
          seriestype = :scatter,
          alpha = 0.35,
          thickness_scaling = 1.35,
          markersize = 3,
          markerstrokewidth = 1,
          markerstrokecolor = :red,
          markershape = :utriangle,
          markercolor = :red2,
          label = "Test forces")
    plot!(dpi = 300,
          label = "",
          xlabel = "F DFT | eV/Å",
          ylabel = "F predicted | eV/Å")
    p = plot!(r0:rs:r1,
              r0:rs:r1,
              label = "", 
              color = :red)
    return p
end


"""
    plot_cos(
        f_true,
        f_pred
    )

`f_true`: vector of true forces
`f_pred`: vector of predicted forces

Returns a plot with the cosine or correlation of the forces.

"""
function plot_cos(
    f_true,
    f_pred
)
    f_true_v = collect(eachcol(reshape(f_true, 3, :)))
    f_pred_v = collect(eachcol(reshape(f_pred, 3, :)))
    f_cos = dot.(f_true_v, f_pred_v) ./ (norm.(f_true_v) .* norm.(f_pred_v))
    p = plot(f_cos,
             seriestype = :scatter,
             markersize = 3,
             markerstrokewidth = 1,
             markershape = :circle,
             markercolor = :gray,
             alpha = 0.35,
             dpi = 300,
             thickness_scaling = 1.35,
             label = "",
             xlabel = "F DFT vs F predicted",
             ylabel = "cos(α)")
    return p
end


"""
    plot_err_time(
        res
    )

`res`: Dataframe with fitting error and force time information.

Returns a plot with fitting errors vs force times. Required in hyper-parameter optimization.

"""
function plot_err_time(res)
    e_mae = res[!, :e_mae]
    f_mae = res[!, :f_mae]
    times = res[!, :time_us]
    plot(times,
         e_mae,
         seriestype = :scatter,
         alpha = 0.55,
         thickness_scaling = 1.35,
         markersize = 3,
         markerstrokewidth = 1,
         markerstrokecolor = :black,
         markershape = :circle,
         markercolor = :gray,
         label = "MAE(E_Pred, E_DFT) | eV/atom")
    plot!(times,
          f_mae,
          seriestype = :scatter,
          alpha = 0.55,
          thickness_scaling = 1.35,
          markersize = 3,
          markerstrokewidth = 1,
          markerstrokecolor = :red,
          markershape = :utriangle,
          markercolor = :red2,
          label = "MAE(F_Pred, F_DFT) | eV/Å")
    plot!(dpi = 300,
          label = "",
          xlabel = "Time per force per atom | µs",
          ylabel = "MAE")
end

