import Optimization 
import Flux
using Printf

abstract type AbstractLearningProblem end
export learn!, LinearProblem, UnivariateLinearProblem, CovariateLinearProblem

include("learn.jl")
include("linear.jl")

