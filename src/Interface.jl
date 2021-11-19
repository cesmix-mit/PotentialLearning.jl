################################################################################
#
#    Interface.jl
#
################################################################################

using GalacticOptim, Optim
using BlackBoxOptim


# Abstract types ###############################################################

"""
    LearningProblem{D, T}
    
"""
abstract type LearningProblem{D, T} end

"""
    LearningOptimizer{D, T}
    
"""
abstract type LearningOptimizer{D, T} end


# Optimizers ###################################################################

"""
    SDPOpt{D, T}
    
Semidefinite program (SDP) optimizer
"""
struct SDPOpt{D, T} <: LearningOptimizer{D, T} end

"""
    LeastSquaresOpt{D, T}
    
Least squares optimizer
"""
struct LeastSquaresOpt{D, T} <: LearningOptimizer{D, T} end

"""
    QRLinearOpt{D, T}
    
QR optimizer
"""
struct QRLinearOpt{D, T} <: LearningOptimizer{D, T} end

"""
    NelderMeadOpt{D, T}
    
Nelder mead optimizer
"""
struct NelderMeadOpt{D, T} <: LearningOptimizer{D, T}
    maxiters
end


# Learning and loss functions ##################################################

"""
    loss(params::Vector{T}, opt::LearningOptimizer{D, T})
    
`params`: parameters to be fitted
`opt`: learning optimizer
"""
function loss(params::Vector{T}, opt::LearningOptimizer{D, T}) end

"""
    learn(lp::LearningProblem{D, T}, opt::LearningOptimizer{D, T})
    
`lp`: learning problem
`opt`: learning optimizer
"""
function learn(lp::LearningProblem{D, T}, opt::LearningOptimizer{D, T}) end

"""
    learn(lp::LearningProblem{D, T}, opt::LearningOptimizer{D, T}, loss)
    
`lp`: learning problem
`opt`: learning optimizer
`loss`: loss function
"""
function learn(lp::LearningProblem{D, T}, opt::LearningOptimizer{D, T}, loss) end

