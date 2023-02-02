"""
    AbstractLearningProblem 

Abstract type to define different types of LearningProblems. There are currently two subtypes: LearningProblem (generic) and LinearProblem (Univariate and Covariate). 
"""
# abstract type AbstractLearningProblem end
export learn!, LinearProblem, UnivariateLinearProblem, CovariateLinearProblem

include("learn.jl")
include("linear.jl")

