abstract type AbstractLearningProblem end

export learn!, get_all_energies, get_all_forces, LBasisPotentialExt

include("InteratomicBasisPotentialsExtension.jl")

include("general-learning-problem.jl")
include("general-learn.jl")

include("linear-learning-problem.jl")
include("mle-linear-learn.jl")
include("wls-linear-learn.jl") # Default learning algorithms

include("utils.jl")
