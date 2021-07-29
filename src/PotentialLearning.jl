################################################################################
#
#    Module PotentialLearning.jl
#
################################################################################

module PotentialLearning

using GalacticOptim, Optim
using BlackBoxOptim
using Printf

export get_conf_params, get_dft_data, learn, validate, SNAP_LAMMPS

abstract type PotentialLearningProblem end

include("./Potentials/Potentials.jl")
include("InputLoading.jl")
include("LearningData.jl")
include("SNAP-LAMMPS.jl")

"""
    learn(p::Potential, dft_training_data::Vector{Float64}, params::Dict)

Fit the potentials, forces, and stresses against the DFT data using the
configuration parameters.
"""
function learn(p::PotentialLearningProblem, params::Dict)
    if params["solver"] == "\\"
        p.β = p.A \ p.b
    elseif params["solver"] == "NelderMead"
        β0 = zeros(length(p.A[1,:]))
        prob = GalacticOptim.OptimizationProblem( (x, pars) -> error(x, p), β0, [])
        p.β = GalacticOptim.solve(prob, NelderMead(), maxiters=500)
    elseif params["solver"] == "BBO"
        β0 = zeros(p.cols)
        lb0 = -0.5ones(p.cols)
        ub0 = 0.5ones(p.cols)
        prob = GalacticOptim.OptimizationProblem( (x, pars) -> error(x, p), β0, [],
                                                  ub = ub0, lb = lb0)
        p.β = solve(prob, BBO(); maxiters=500)
    end
end

"""
    validate_potentials(p::PotentialLearningProblem,
                        dft_validation_data::Vector{Float64}, params::Dict)

Validate trained potentials, forces, and stresses.
"""
function validate(p::PotentialLearningProblem, dft_validation_data::Vector{Float64},
                  params::Dict)
    fit_forces = params["fit_forces"]
    rcut = params["rcut"]
    no_atoms_per_conf = params["no_atoms_per_conf"]
    no_train_atomic_conf = params["no_train_atomic_conf"]
    no_atomic_conf = params["no_atomic_conf"]
    no_val_atomic_conf = no_atomic_conf - no_train_atomic_conf
    rel_errors = []
    
    io = open("energy_validation.csv", "w");
    line = @sprintf("Configuration, DFT Potential Energy, Fitted Potential Energy, Relative Error\n")
    write(io, line)
    for j in 1:no_val_atomic_conf
        p_dft = dft_validation_data[j]
        p_fitted = potential_energy(p, j + no_train_atomic_conf, params)
        rel_error = abs(p_dft - p_fitted) / p_dft
        push!(rel_errors, rel_error)
        line = @sprintf("%d, %0.2f, %0.2f, %0.2f\n", j+ no_train_atomic_conf, p_dft, p_fitted, rel_error)
        write(io, line)
    end
    close(io)
    
    if fit_forces
        io = open("force_validation.csv", "w");
        line = @sprintf("Configuration, No. Atom, DFT Force, Fitted Force, Relative Error\n")
        write(io, line)
        for j in 1:no_val_atomic_conf
            fitted_forces = forces(p, j + no_train_atomic_conf, params)
            for k in 1:no_atoms_per_conf
                f_dft = Force(dft_validation_data[k*3-2],
                              dft_validation_data[k*3-1],
                              dft_validation_data[k*3])
                rel_error = norm(f_dft - fitted_forces[k]) / norm(f_dft)
                push!(rel_errors, rel_error)
                line = @sprintf("%d, %d, %0.2f %0.2f %0.2f, %0.2f %0.2f %0.2f, %0.2f\n",
                        j+ no_train_atomic_conf, k, f_dft[1], f_dft[2], f_dft[3],
                        fitted_forces[k][1], fitted_forces[k][2], fitted_forces[k][3],
                        rel_error)
                write(io, line)
            end
        end
        close(io)
    end
    
    # TODO: validate stresses
    
    return maximum(rel_errors)
end

end

