abstract type AbstractLearningProblem end

export learn!, get_all_energies, get_all_forces

include("general-learning-problem.jl")
include("linear-learning-problem.jl")
include("linear-learn.jl")

