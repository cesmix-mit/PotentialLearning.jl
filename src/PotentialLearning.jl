################################################################################
#
#    Module PotentialLearning.jl
#
################################################################################

module PotentialLearning

using GalacticOptim, Optim
using BlackBoxOptim
using Printf

export get_conf_params, generate_data, learning_problem, learn, validate

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
    validate(p::PotentialLearningProblem, val_data::Vector{Float64}, params::Dict)

Validates trained potentials, forces, and stresses.
"""
function validate(p::PotentialLearningProblem, val_data::Vector{Float64}, params::Dict)
    fit_forces = params["global"]["fit_forces"]
    rcut = params["global"]["rcut"]
    no_atoms_per_conf = params["global"]["no_atoms_per_conf"]
    no_train_atomic_conf = params["global"]["no_train_atomic_conf"]
    no_atomic_conf = params["global"]["no_atomic_conf"]
    no_val_atomic_conf = no_atomic_conf - no_train_atomic_conf
    rel_errors = []
    
    io = open("energy_validation.csv", "w");
    line = @sprintf("Configuration, Potential Energy, Fitted Potential Energy, Relative Error\n")
    write(io, line)
    for j in 1:no_val_atomic_conf
        p_val = val_data[j]
        p_fitted = potential_energy(p, j + no_train_atomic_conf)
        rel_error = abs(p_val - p_fitted) / p_val
        push!(rel_errors, rel_error)
        line = @sprintf("%d, %0.2f, %0.2f, %0.2f\n",
                         j+ no_train_atomic_conf, p_val, p_fitted, rel_error)
        write(io, line)
    end
    close(io)
    
    if fit_forces
        io = open("force_validation.csv", "w");
        line = @sprintf("Configuration, No. Atom, Force, Fitted Force, Relative Error\n")
        write(io, line)
        for j in 1:no_val_atomic_conf
            fitted_forces = forces(p, j + no_train_atomic_conf)
            for k in 1:no_atoms_per_conf
                f_val = Force(val_data[k*3-2],
                              val_data[k*3-1],
                              val_data[k*3])
                rel_error = norm(f_val - fitted_forces[k]) / norm(f_val)
                push!(rel_errors, rel_error)
                line = @sprintf("%d, %d, %0.2f %0.2f %0.2f, %0.2f %0.2f %0.2f, %0.2f\n",
                        j+ no_train_atomic_conf, k, f_val[1], f_val[2], f_val[3],
                        fitted_forces[k][1], fitted_forces[k][2],
                        fitted_forces[k][3], rel_error)
                write(io, line)
            end
        end
        close(io)
    end
    
    # TODO: validate stresses
    
    return maximum(rel_errors)
end

end

