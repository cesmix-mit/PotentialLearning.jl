
export SmallSNAPLP, hyperparam_loss, β_loss, learn

# Learning problem #############################################################


"""
    SmallSNAPLP
    
SNAP learning problem for small systems
"""
struct SmallSNAPLP <: LearningProblem
   
   # Training #################################
   snap::SNAP         # SNAP potential
   A::Matrix       # SNAP matrix
   y::Vector       # DFT and reference data
   ATA::Matrix     # A^T * A
   ATy::Vector     # A^T * y
   Q::Diagonal     # Covariance
   invQ::Diagonal  # inverse of Q
   training_dft_data::SmallESData
   training_ref_data::SmallESData
    
   # Validation ###############################
   A_val::Matrix   # SNAP matrix
   y_val::Vector   # DFT and reference data
   validation_dft_data::SmallESData
   validation_ref_data::SmallESData
   
end

function SmallSNAPLP(snap::SNAP, atomic_confs::Vector, data::SmallESData;
                     trainingsize = 0.8, fit = [:e, :f, :s]) where 
    # TODO: @views?
    
    # Training #############################################################
    m = floor(Int, length(atomic_confs) * trainingsize)
    AA = get_snap(atomic_confs[1:m], snap) #TODO: get_snap should allow me to chose what to fit (e, f, or s)
    n_e = m; n_f = n_e + m * length(atomic_confs[1].Atoms) * 3
    A = []
    if :e in fit A = AA[1:n_e, :] end
    if :f in fit A = vcat(A, AA[n_e+1:n_f, :]) end
    if :s in fit A = vcat(A, AA[n_f+1:end, :]) end
    ATA = A' * A
    Q = Diagonal(ones(size(A)[1])); invQ = inv(Q)
    training_dft_data = SmallESData(data.energies[1:m],
                                       data.forces[1:m],
                                       data.stresses[1:m])
    # TODO: add ZBL, see PotentialLearning.jl in master
    training_ref_data = SmallESData(data.energies[1:m] * 0.01,
                                       data.forces[1:m] * 0.01,
                                       data.stresses[1:m] * 0.01)
    y = []
    if :e in fit
        y = linearize(training_dft_data.energies) -
            linearize(training_ref_data.energies)
    end
    if :f in fit
        append!(y, linearize(training_dft_data.forces) -
                   linearize(training_ref_data.forces))
    end
    if :s in fit
        append!(y, linearize(training_dft_data.stresses) -
                   linearize(training_ref_data.stresses))
    end
    ATy = A' * y

    # Validation ###########################################################
    m_val = length(atomic_confs[m+1:end])
    AA_val = get_snap(atomic_confs[m+1:end], snap)
    n_e = m_val; n_f = n_e + m_val * length(atomic_confs[1].Atoms) * 3
    A_val = []
    if :e in fit A_val = AA_val[1:n_e, :] end
    if :f in fit A_val = vcat(A_val, AA_val[n_e+1:n_f, :]) end
    if :s in fit A_val = vcat(A_val, AA_val[n_f+1:end, :]) end
    validation_dft_data = SmallESData(data.energies[m+1:end],
                                         data.forces[m+1:end],
                                         data.stresses[m+1:end])
    # TODO: add ZBL, see PotentialLearning.jl in master
    validation_ref_data = SmallESData(data.energies[m+1:end] * 0.01,
                                         data.forces[m+1:end] * 0.01,
                                         data.stresses[m+1:end] * 0.01)
    y_val = []
    if :e in fit
        y_val = linearize(validation_dft_data.energies) -
                linearize(validation_ref_data.energies)
    end
    if :f in fit
        append!(y_val, linearize(validation_dft_data.forces) -
                       linearize(validation_ref_data.forces))
    end
    if :s in fit
        append!(y_val, linearize(validation_dft_data.stresses) -
                       linearize(validation_ref_data.stresses))
    end

    SmallSNAPLP(snap, A, y, ATA, ATy, Q, invQ, training_dft_data,
                training_ref_data, A_val, y_val, validation_dft_data,
                validation_ref_data)
end



# Hyper-parameter optimization #################################################

"""
    hyperparam_loss(hyperparams::Vector, lp::SmallSNAPLP)
    
SNAP hyper-parameter loss function
"""
function hyperparam_loss(hyperparams::Vector, lp::SmallSNAPLP) where 
#   return  ...
end


"""
    learn(lp::SmallSNAPLP, loss, s::SDPOpt)
    
Learning function: using SDPOpt to optimize the hyper-parameters
"""
function learn(lp::SmallSNAPLP, loss, s::SDPOpt) where 
#   ...hyperparam_loss...
#   lp.snap.rcutfac = ...
#   lp.snap.twojmax = ...
end


# β-parameter optimization #####################################################

"""
    learn(lp::SmallSNAPLP, s::LeastSquaresOpt)
    
Learning function: using normal equations and qr decomposition to learn main parameters
"""
function learn(lp::SmallSNAPLP, s::LeastSquaresOpt) where 
   lp.snap.β = (lp.A' * lp.A) \ (lp.A' * lp.y)
end

"""
    learn(lp::SmallSNAPLP, s::QRLinearOpt)
    
Learning function: using QR decomposition to learn main parameters
"""
function learn(lp::SmallSNAPLP, s::QRLinearOpt) where 
   lp.snap.β = lp.A \ lp.y # or... qr(A,Val(true)) \ y
end

"""
    β_loss(β::Vector, lp::SmallSNAPLP)
    
Loss function: using NelderMeadOpt from GalacticOptim to optimize the parameters
"""
function β_loss(β::Vector, lp::SmallSNAPLP) where 
   e = lp.A * β - lp.y
   return  transpose(e) * inv(lp.Q) * e
end

"""
    learn(lp::SmallSNAPLP, loss, s::NelderMeadOpt)
    
Learning function: using NelderMeadOpt from GalacticOptim to optimize the parameters
"""
function learn(lp::SmallSNAPLP, loss, s::NelderMeadOpt) where 
   β0 = zeros(length(lp.A[1,:]))
   prob = GalacticOptim.OptimizationProblem((x, pars)->loss(x, lp), β0, [])
   lp.snap.β = GalacticOptim.solve(prob, NelderMead(), maxiters=s.maxiters)
end


