
export SmallSNAPLP, hyperparam_loss, β_loss, learn

# Learning problem #############################################################


"""
    SmallSNAPLP{D, T}
    
SNAP learning problem for small systems
"""
struct SmallSNAPLP{D, T} <: LearningProblem{D, T}
   # SNAP potential
   snap::SNAP
  
   # SNAP matrix (depends on the hyper-parameters)
   A::Matrix{T}
   ATA::Matrix{T} # A^T * A, optional
 
   # Covariance
   Q::Diagonal{T}
   invQ::Diagonal{T} # inverse of A, optional
  
   # Input data
   y::Vector{T} # linearize(training_dft_data) - linearize(training_ref_data)
   ATy::Matrix{T} # A^T * A, optional
   training_dft_data::SmallESData{D}
   training_ref_data::SmallESData{D}
   validation_dft_data::SmallESData{D}
   validation_ref_data::SmallESData{D}
end

function SmallSNAPLP(snap::SNAP, data::SmallESData)
    
#    A = get_snap(inter_pot_atomic_confs, snap)
#    ATA = A' * A
#    n = size(A)[1]
#    Q = Diagonal(ones(n))
#    invQ = inv(Q)
#    y = 
#    SNAP(snap, A, )
end


# Hyper-parameter optimization #################################################

"""
    hyperparam_loss(hyperparams::Vector{T}, lp::SmallSNAPLP{D, T})
    
SNAP hyper-parameter loss function
"""
function hyperparam_loss(hyperparams::Vector{T}, lp::SmallSNAPLP{D, T}) where {D, T}
#   return  ...
end


"""
    learn(lp::SmallSNAPLP{T}, loss, s::SDPOpt{T})
    
Learning function: using SDPOpt to optimize the hyper-parameters
"""
function learn(lp::SmallSNAPLP{T}, loss, s::SDPOpt{T}) where {T}
#   ...hyperparam_loss...
#   lp.snap.rcutfac = ...
#   lp.snap.twojmax = ...
end


# β-parameter optimization #####################################################

"""
    learn(lp::SmallSNAPLP{T}, s::LeastSquaresOpt{T})
    
Learning function: using normal equations and qr decomposition to learn main parameters
"""
function learn(lp::SmallSNAPLP{T}, s::LeastSquaresOpt{T}) where {T}
   lp.snap.β = (lp.A' * lp.A) \ (lp.A' * lp.y)
end

"""
    learn(lp::SmallSNAPLP{T}, s::QRLinearOpt{T})
    
Learning function: using QR decomposition to learn main parameters
"""
function learn(lp::SmallSNAPLP{T}, s::QRLinearOpt{T}) where {T}
   lp.snap.β = lp.A \ lp.y # or... qr(A,Val(true)) \ y
end

"""
    β_loss(β::Vector{T}, lp::SmallSNAPLP{D, T})
    
Loss function: using NelderMeadOpt from GalacticOptim to optimize the parameters
"""
function β_loss(β::Vector{T}, lp::SmallSNAPLP{D, T}) where {D, T}
   e = lp.A * β - lp.y
   return  transpose(e) * inv(lp.Q) * e
end

"""
    learn(lp::SmallSNAPLP{T}, loss, s::NelderMeadOpt{T})
    
Learning function: using NelderMeadOpt from GalacticOptim to optimize the parameters
"""
function learn(lp::SmallSNAPLP{T}, loss, s::NelderMeadOpt{T}) where {T}
   β0 = zeros(length(lp.A[1,:]))
   prob = GalacticOptim.OptimizationProblem((x, pars)->loss(x, lp), β0, [])
   lp.β = GalacticOptim.solve(prob, NelderMead(), maxiters=s.maxiters)
end


