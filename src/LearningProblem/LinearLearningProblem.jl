struct LinearProblem{D, T} <: LearningProblem{D, T}
    descriptors :: Vector{Vector{D}}
end