################################################################################
#
#    Interface.jl
#
################################################################################

using ElectronicStructure, InteratomicPotentials
using LinearAlgebra, GalacticOptim, Optim, BlackBoxOptim

export LearningProblem, LearningOptimizer
export SDPOpt, LeastSquaresOpt, QRLinearOpt, NelderMeadOpt
export loss, learn


# Abstract types ###############################################################

"""
    LearningProblem
    
"""
abstract type LearningProblem end

"""
    LearningOptimizer
    
"""
abstract type LearningOptimizer end


# Optimizers ###################################################################

"""
    SDPOpt
    
Semidefinite program (SDP) optimizer
"""
struct SDPOpt <: LearningOptimizer end

"""
    LeastSquaresOpt
    
Least squares optimizer
"""
struct LeastSquaresOpt <: LearningOptimizer end

"""
    QRLinearOpt
    
QR optimizer
"""
struct QRLinearOpt <: LearningOptimizer end

"""
    NelderMeadOpt
    
Nelder mead optimizer
"""
struct NelderMeadOpt <: LearningOptimizer
    maxiters
end


# Learning and loss functions ##################################################

"""
    loss(params::Vector, opt::LearningOptimizer)
    
`params`: parameters to be fitted
`opt`: learning optimizer
"""
function loss(params::Vector, opt::LearningOptimizer) end

"""
    learn(lp::LearningProblem, opt::LearningOptimizer)
    
`lp`: learning problem
`opt`: learning optimizer
"""
function learn(lp::LearningProblem, opt::LearningOptimizer) end

"""
    learn(lp::LearningProblem, opt::LearningOptimizer, loss)
    
`lp`: learning problem
`opt`: learning optimizer
`loss`: loss function
"""
function learn(lp::LearningProblem, opt::LearningOptimizer, loss) end

