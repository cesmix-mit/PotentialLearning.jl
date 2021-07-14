################################################################################
#
#    Module PotentialLearning.jl
#
################################################################################

module PotentialLearning

using GalacticOptim, Optim, Printf

export load_learning_params, load_dft_data, learn, validate, SNAP_LAMMPS

include("Utils.jl")
include("InputLoading.jl")
include("EmpiricalPotentials.jl")
include("SNAP-LAMMPS.jl")

function learn(p, dft_training_data::Vector{Float64}, learning_params::Dict)
    p.b = dft_training_data
    
    if learning_params["solver"] == "\\"
        p.β = p.A \ p.b
    else
        β0 = zeros(p.rows)
        prob = OptimizationProblem(error, β0, [], p)
        p.β = solve(prob, NelderMead())
    end
end

function validate(p, dft_validation_data::Vector{Float64}, learning_params::Dict)
    rcut = learning_params["rcut"]
    rows = learning_params["rows"]
    rel_errors = []
    @printf("Potential Energy, Fitted Potential Energy, Relative Error\n")
    for (j, p_dft) in enumerate(dft_validation_data)
        p_fitted = potential_energy(learning_params, j + rows, p)
        rel_error = abs(p_dft - p_fitted) / p_dft
        push!(rel_errors, rel_error)
        @printf("%0.2f, %0.2f, %0.2f\n", p_dft, p_fitted, rel_error)
    end
    return maximum(rel_errors)
end

end

