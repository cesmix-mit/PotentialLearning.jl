################################################################################
#
#    Module PotentialLearning.jl
#
################################################################################

module PotentialLearning

include("Utils.jl")
include("InputLoading.jl")
include("EmpiricalPotentials.jl")
include("SNAP-LAMMPS.jl")

using GalacticOptim, Optim, Printf


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

#    @printf("Potential Energy, Fitted Potential Energy, Error (%%)\n")
#    for (j, rs) in enumerate(p.dft_validation_data)
#        p_GaN_model = potential_energy(rs, rcut, p)
#        p_fitted = potential_energy(path, j + n, p)
#        rel_error = abs(p_GaN_model - p_fitted) / p_GaN_model * 100.
#        @test rel_error < 10.0 
#        @printf("%0.2f, %0.2f, %0.2f\n", p_GaN_model, p_fitted, rel_error)
#    end

    return 0.01
end

end

