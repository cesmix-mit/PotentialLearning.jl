################################################################################
#
#    Module PotentialLearning.jl
#
################################################################################

module PotentialLearning

using GalacticOptim, Optim
using BlackBoxOptim
using Printf

export load_conf_params, load_dft_data, learn, validate_potentials, SNAP_LAMMPS

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
    elseif params["solver"] == "NelderMead"
        β0 = zeros(length(p.A[1,:]))
        prob = GalacticOptim.OptimizationProblem( (x, pars) -> error(x, p), β0, [])
        p.β = GalacticOptim.solve(prob, NelderMead(), maxiters=500)
    elseif params["solver"] == "BBO"
        β0 = zeros(length(p.A[1,:]))
        lb0 = -ones(length(p.A[1,:]))
        ub0 = ones(length(p.A[1,:]))
        prob = GalacticOptim.OptimizationProblem( (x, pars) -> error(x, p), β0, [],
                                                  ub = ub0, lb = lb0)
        p.β = solve(prob, BBO(), maxiters=500)
    end
end

"""
    validate(p::Potential, dft_validation_data::Vector{Float64}, params::Dict)
    
Validate trained potentials, forces, and stresses.
"""
function validate_potentials(p::Potential, dft_validation_data::Vector{Float64}, params::Dict)
    rcut = params["rcut"]
    no_train_atomic_conf = params["no_train_atomic_conf"]
    no_val_energies = params["no_atomic_conf"] - params["no_train_atomic_conf"]
    rel_errors = []
    @printf("Potential Energy, Fitted Potential Energy, Relative Error\n")
    for j in 1:no_val_energies
        p_dft = dft_validation_data[j]
        p_fitted = potential_energy(params, j + no_train_atomic_conf, p)
        rel_error = abs(p_dft - p_fitted) / p_dft
        push!(rel_errors, rel_error)
        @printf("%0.2f, %0.2f, %0.2f\n", p_dft, p_fitted, rel_error)
    end
    return maximum(rel_errors)
end

end

