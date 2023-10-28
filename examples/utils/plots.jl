# TODO: units are hardcoded

"""
    plot_energy(e_pred, e_true)
    
`e_pred`: vector of predicted energies
`e_true`: vector of true energies

Returns an energy plot.

"""
function plot_energy(e_pred, e_true)
    r0 = minimum(e_true); r1 = maximum(e_true); rs = (r1-r0)/10
    plot( e_true, e_pred, seriestype = :scatter, 
          markersize=5, markerstrokewidth=0, markershape=:circle, markercolor=:gray,
          label="", xlabel = "E DFT | eV", ylabel = "E predicted | eV")
    p = plot!( r0:rs:r1, r0:rs:r1, label="")
    return p
end


"""
    plot_energy(e_pred, e_true)
    
`e_train_pred`: vector of predicted energies of training dataset
`e_train_true`: vector of true energies of training dataset
`e_test_pred`: vector of predicted energies of test dataset
`e_test_true`: vector of true energies of test dataset

Returns an energy plot.

"""
function plot_energy(
    e_train_pred,
    e_train_true,
    e_test_pred,
    e_test_true
)
    es =[e_train_pred; e_train_true; e_test_pred; e_test_true]
    r0 = minimum(es); r1 = maximum(es); rs = (r1-r0)/10
    plot(e_train_true, e_train_pred, seriestype=:scatter, label="Training energies",
         markersize=5, markerstrokewidth=0, markershape=:circle, markercolor=:gray)
    plot!(e_test_true, e_test_pred, seriestype=:scatter,  label="Test energies",
          markersize=5, markerstrokewidth=0, markershape=:utriangle, markercolor=:black)
    plot!(label="", xlabel = "E DFT | eV", ylabel = "E predicted | eV")
    p = plot!( r0:rs:r1, r0:rs:r1, label="", linecolor=:gray)
    return p
end


"""
    plot_forces(f_pred, f_true)
    
`f_pred`: vector of predicted forces
`f_true`: vector of true forces

Returns a force plot.

"""
function plot_forces(f_pred, f_true)
    r0 = floor(minimum(f_true)); r1 = ceil(maximum(f_true))
    plot( f_true, f_pred, seriestype = :scatter,
          markersize=5, markerstrokewidth=0, markershape=:circle, markercolor=:gray,
          label="", xlabel = "F DFT | eV/Å", ylabel = "F predicted | eV/Å", 
          xlims = (r0, r1), ylims = (r0, r1))
    p = plot!( r0:r1, r0:r1, label="")
    return p
end


"""
    plot_cos(f_pred, f_true)
    
`f_pred`: vector of predicted forces
`f_true`: vector of true forces

Returns a plot with the cosine or correlation of the forces.

"""
function plot_cos(f_pred, f_true)
    f_pred_v = collect(eachcol(reshape(f_pred, 3, :)))
    f_true_v = collect(eachcol(reshape(f_true, 3, :)))
    f_cos = dot.(f_true_v, f_pred_v) ./ (norm.(f_true_v) .* norm.(f_pred_v))
    p = plot( f_cos, seriestype = :scatter, markersize=5, markerstrokewidth=0,
              markershape=:circle, markercolor=:gray,
              label="", xlabel = "F DFT vs F predicted", ylabel = "cos(α)")
    return p
end

