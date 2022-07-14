module PotentialLearning

include("Interface.jl")
include("IO/Input.jl")
include("IO/Load-extxyz.jl")
include("IO/Utils.jl")
include("Learning/NNBasisPotential.jl") # TODO: Add to InteratomicPotentials.jl/InteratomicBasisPotentials.jl
include("Learning/Losses.jl") 
include("Learning/Learning.jl")
include("PostProc/Metrics.jl")
include("PostProc/Plots.jl")

end
