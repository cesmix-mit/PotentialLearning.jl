
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
   invQ::Diagonal{T} # inverse of Q, optional
  
   # Input data
   training_dft_data::SmallESData{D}
   training_ref_data::SmallESData{D}
   validation_dft_data::SmallESData{D}
   validation_ref_data::SmallESData{D}
   y::Vector{T} # linearize(training_dft_data) - linearize(training_ref_data)
   y_val::Vector{T} # linearize(validation_dft_data) - linearize(validation_ref_data)
   ATy::Vector{T} # A^T * y, optional
   
end

function SmallSNAPLP(snap::SNAP, atomic_confs::Vector, data::SmallESData{D}; training_prop = 0.8) where {D}
    m = floor(Int, length(data.energies) * training_prop)
    n = floor(Int, length(data.forces) * training_prop)
    
    A = get_snap(atomic_confs[1:m], snap)
    ATA = A' * A
    nrows = size(A)[1]
    Q = Diagonal(ones(nrows))
    invQ = inv(Q)
    
    training_dft_data = SmallESData{D}(data.energies[1:m],
                                       data.forces[1:n],
                                       data.stresses[1:m]) # TODO: @views?
    validation_dft_data = SmallESData{D}(data.energies[m+1:end],
                                         data.forces[n+1:end],
                                         data.stresses[m+1:end]) # TODO: @views?
    training_ref_data = training_dft_data # TODO: add ZBL, see PotentialLearning.jl in master
    validation_ref_data = validation_dft_data # TODO: add ZBL, see PotentialLearning.jl in master
    
    y = linearize(training_dft_data) #- linearize(training_ref_data)
    y_val = linearize(validation_dft_data) #- linearize(validation_ref_data)
    ATy = A' * y

    
    SmallSNAPLP(snap, A, ATA, Q, invQ, training_dft_data, training_ref_data,
                validation_dft_data, validation_ref_data, y, y_val, ATy)
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
   lp.snap.β = GalacticOptim.solve(prob, NelderMead(), maxiters=s.maxiters)
end


