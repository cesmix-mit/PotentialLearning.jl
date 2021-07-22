################################################################################
#
#    Module PotentialLearning.jl
#
################################################################################

module PotentialLearning

using GalacticOptim, Optim, Printf

export load_conf_params, load_dft_data, learn, validate, SNAP_LAMMPS

include("EmpiricalPotentials.jl")
include("SNAP-LAMMPS.jl")
include("InputLoading.jl")


"""
    learn(p::Potential, dft_training_data::Vector{Float64}, params::Dict)

Fit the potentials, forces, and stresses against the DFT data using the
configuration parameters.
"""
function learn(p::Potential, dft_training_data::Vector{Float64}, params::Dict)
    p.b = dft_training_data
    
    if params["solver"] == "\\"
        p.β = p.A \ p.b
    else
        β0 = zeros(p.no_train_atomic_conf)
        prob = OptimizationProblem(error, β0, [], p)
        p.β = solve(prob, NelderMead())
    end
end

"""
    validate(p::Potential, dft_validation_data::Vector{Float64}, params::Dict)
    
Validate trained potentials, forces, and stresses.
"""
function validate(p::Potential, dft_validation_data::Vector{Float64}, params::Dict)
    rcut = params["rcut"]
    no_train_atomic_conf = params["no_train_atomic_conf"]
    rel_errors = []
    @printf("Potential Energy, Fitted Potential Energy, Relative Error\n")
    for (j, p_dft) in enumerate(dft_validation_data)
        p_fitted = potential_energy(params, j + no_train_atomic_conf, p)
        rel_error = abs(p_dft - p_fitted) / p_dft
        push!(rel_errors, rel_error)
        @printf("%0.2f, %0.2f, %0.2f\n", p_dft, p_fitted, rel_error)
    end
    return maximum(rel_errors)
end

end

