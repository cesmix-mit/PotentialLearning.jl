################################################################################
#
#    Module PotentialLearning.jl
#
################################################################################

module PotentialLearning

using GalacticOptim, Optim
using BlackBoxOptim
using Printf

export get_conf_params, get_dft_data, learn, validate_potentials, SNAP_LAMMPS

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

Validate trained potentials.
"""
function validate_potentials(p::PotentialLearningProblem,
                             dft_validation_data::Vector{Float64}, params::Dict)
    rcut = params["rcut"]
    no_train_atomic_conf = params["no_train_atomic_conf"]
    no_val_energies = params["no_atomic_conf"] - params["no_train_atomic_conf"]
    rel_errors = []
    @printf("Potential Energy, Fitted Potential Energy, Relative Error\n")
    for j in 1:no_val_energies
        p_dft = dft_validation_data[j]
        p_fitted = potential_energy(p, j + no_train_atomic_conf, params)
        rel_error = abs(p_dft - p_fitted) / p_dft
        push!(rel_errors, rel_error)
        @printf("%0.2f, %0.2f, %0.2f\n", p_dft, p_fitted, rel_error)
    end
    return maximum(rel_errors)
end

#"""
#    validate_forces(p::PotentialLearningProblem,
#                    dft_validation_data::Vector{Float64}, params::Dict)

#Validate trained forces.
#"""
#function validate_forces(p::PotentialLearningProblem,
#                         dft_validation_data::Vector{Float64}, params::Dict)
#    rcut = params["rcut"]
#    no_train_atomic_conf = params["no_train_atomic_conf"]
#    rel_errors = []
#    @printf("Force, Fitted Force, Relative Error\n")
#    for j in no_train_atomic_conf+1:length(dft_validation_data) # check this when adding stresses
#        f_dft = dft_validation_data[j]
#        f_fitted = force(params, j + no_train_atomic_conf, p) #????
#        rel_error = abs(f_dft - f_fitted) / f_dft
#        push!(rel_errors, rel_error)
#        @printf("%0.2f, %0.2f, %0.2f\n", p_dft, p_fitted, rel_error)
#    end
#    return maximum(rel_errors)
#end

end

