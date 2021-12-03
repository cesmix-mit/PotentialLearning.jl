
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

function SmallSNAPLP(snap::SNAP, atomic_confs::Vector, data::SmallESData{D};
                     trainingsize = 0.8, fit = [:e, :f, :s]) where {D}

    m = floor(Int, length(atomic_confs) * trainingsize)

    n_e = floor(Int, length(data.energies) * trainingsize)
    n_f = n_e + floor(Int, length(data.forces) * length(data.forces[1]) * 3 * trainingsize)
    n_s = n_f + floor(Int, length(data.stresses) * sum([ 1 for j in 1:D for i in j:D ]) * trainingsize)
    
    AA = get_snap(atomic_confs[1:m], snap) #TODO: get_snap should allow me to chose what to fit (e, f, or s)
    A_e = AA[1:n_e, :]
    A_f = AA[n_e+1:n_f, :]
    A_s = AA[n_f+1:n_s, :]
    
    A = []
    if :e in fit A = A_e end
    if :f in fit A = vcat(A, A_f) end
    if :s in fit A = vcat(A, A_s) end
    
    ATA = A' * A
    nrows = size(A)[1]
    Q = Diagonal(ones(nrows))
    invQ = inv(Q)
    
    training_dft_data = SmallESData{D}(data.energies[1:m],
                                       data.forces[1:m],
                                       data.stresses[1:m]) # TODO: @views?
    validation_dft_data = SmallESData{D}(data.energies[m+1:end],
                                         data.forces[m+1:end],
                                         data.stresses[m+1:end]) # TODO: @views?
    training_ref_data = training_dft_data # TODO: add ZBL, see PotentialLearning.jl in master
    validation_ref_data = validation_dft_data # TODO: add ZBL, see PotentialLearning.jl in master

    y = []
    if :e in fit
        y = linearize(training_dft_data.energies) #- linearize(training_ref_data.energies)
        y_val = linearize(validation_dft_data.energies) #- linearize(validation_ref_data.energies)
    end
    if :f in fit
        append!(y, linearize(training_dft_data.forces)) #- linearize(training_ref_data.forces)
        append!(y_val, linearize(validation_dft_data.forces)) #- linearize(validation_ref_data.forces)
    end
    if :s in fit
        append!(y, linearize(training_dft_data.stresses)) #- linearize(training_ref_data.stresses)
        append!(y_val, linearize(validation_dft_data.stresses)) #- linearize(validation_ref_data.stresses)
    end

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


