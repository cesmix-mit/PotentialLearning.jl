################################################################################
#
#    Module PotentialLearning.jl
#
################################################################################

module PotentialLearning

using GalacticOptim, Optim
using BlackBoxOptim
using Printf

export get_conf_params, generate_data, get_dft_data, learning_problem, learn,
       error_metrics, potential_energy, forces

abstract type PotentialLearningProblem end

include("./Potentials/Potentials.jl")
include("InputLoading.jl")
include("LearningData.jl")

"""
    learning_problem( dft_train_data::Vector{Float64},
                      ref_train_data::Vector{Float64},
                      params::Dict)

Creates a potential learning problem.
"""
function learning_problem(dft_train_data::Vector{Float64},
                          ref_train_data::Vector{Float64},
                          params::Dict)
    lp = Symbol(params["global"]["potential"])
    return @eval $lp($dft_train_data, $ref_train_data, $params)
end

"""
    learn(p::PotentialLearningProblem, params::Dict)

Fits the potentials, forces, and stresses against the DFT and reference data
using the configuration parameters.
"""
function learn(p::PotentialLearningProblem, params::Dict)
    if params["solver"]["name"] == "\\"
        p.β = p.A \ p.b
    elseif params["solver"]["name"] == "NelderMead"
        β0 = zeros(length(p.A[1,:]))
        prob = GalacticOptim.OptimizationProblem( (x, pars) -> error(x, p), β0, [])
        p.β = GalacticOptim.solve(prob, NelderMead(), maxiters=500)
    elseif params["solver"]["name"] == "BBO"
        β0 = zeros(p.cols)
        lb0 = -0.5ones(p.cols)
        ub0 = 0.5ones(p.cols)
        prob = GalacticOptim.OptimizationProblem( (x, pars) -> error(x, p), β0, [],
                                                  ub = ub0, lb = lb0)
        p.β = solve(prob, BBO(); maxiters=500)
    end
end

"""
    error_metrics(p::PotentialLearningProblem, val_data::Vector{Float64}, params::Dict)

Calculates error metrics: max_rel_error, mae, rmse
"""
function error_metrics(p::PotentialLearningProblem, val_data::Vector{Float64}, params::Dict)
    fit_forces = params["global"]["fit_forces"]
    rcutfac = params["global"]["rcutfac"]
    no_atoms_per_conf = params["global"]["no_atoms_per_conf"]
    no_train_atomic_conf = params["global"]["no_train_atomic_conf"]
    no_atomic_conf = params["global"]["no_atomic_conf"]
    no_val_atomic_conf = no_atomic_conf - no_train_atomic_conf

    global metrics = Dict()
    metrics["energy"] = Dict()
    
    # Energy metrics
    energies = val_data[1:no_val_atomic_conf]
    fitted_energies = [potential_energy(p, j) for j in no_train_atomic_conf+1:no_atomic_conf]
    metrics["energy"]["max_rel_error"] =
                    maximum(abs.(fitted_energies .- energies) ./ fitted_energies)
    metrics["energy"]["mae"] = 
                    sum(abs.(fitted_energies .- energies)) / length(energies)
    metrics["energy"]["rmse"] =
                    sqrt(sum((fitted_energies .- energies).^2) / length(energies))
    
    # Force metrics
    if fit_forces
        forces_ = val_data[no_val_atomic_conf+1:end]
        fitted_forces = linearize([forces(p, j)
                                   for j in no_train_atomic_conf+1:no_atomic_conf])
        metrics["force"] = Dict()
        metrics["force"]["max_rel_error"] =
                    maximum(abs.((fitted_forces .- forces_) ./ forces_))
        metrics["force"]["mae"] = 
                    sum(abs.(fitted_forces .- forces_)) / length(forces_)
        metrics["force"]["rmse"] =
                    sqrt(sum((fitted_forces .- forces_).^2) / length(forces_))
    end

    # TODO: Stress metrics

   return metrics
end

end

