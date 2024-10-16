# Ordinary least squares functions #############################################

"""
function learn!(
    lp::UnivariateLinearProblem,
    α::Real
)

Fit a univariate Gaussian distribution for the equation y = Aβ + ϵ, where β are model coefficients and ϵ ∼ N(0, σ). Fitting is done via SVD on the design matrix, A'*A (formed iteratively), where eigenvalues less than α are cut-off.  
"""
function learn!(
    lp::UnivariateLinearProblem,
    α::Real
)
    # Form design matrices 
    AtA = sum(v * v' for v in lp.iv_data)
    Atb = sum(v * b for (v, b) in zip(lp.iv_data, lp.dv_data))

    Q = pinv(AtA, α)
    lp.β .= Q * Atb
    lp.σ .= std(Atb - AtA * lp.β)
    lp.Σ .= Symmetric(lp.σ[1]^2 * Q)
end


"""
function learn!(
    lp::CovariateLinearProblem,
    α::Real
)

Fit a Gaussian distribution by finding the MLE of the following log probability:
    ℓ(β, σe, σf) = -0.5*(e - A_e *β)'*(e - A_e * β) / σe - 0.5*(f - A_f *β)'*(f - A_f * β) / σf - log(σe) - log(σf)

through an optimization procedure. 
"""
function learn!(
    lp::CovariateLinearProblem,
    α::Real
)
    # Regularizaiton parameter α

    # Does not have analytical solution, use optimization 
    # break into energy and force components
    AtAe = sum(b * b' for b in lp.B)
    Atbe = sum(b * e for (b, e) in zip(lp.B, lp.e))

    AtAf = sum(db * db' for db in lp.dB)
    Atbf = sum(db * f for (db, f) in zip(lp.dB, lp.f))

    f(x, p) =
        -logpdf(MvNormal(p[1] * x[3:end], exp(x[1]) + p[5]), p[2]) -
        logpdf(MvNormal(p[3] * x[3:end], exp(x[2]) + p[5]), p[4])
    g = Optimization.OptimizationFunction(f, Optimization.AutoForwardDiff())

    x0 = [log(lp.σe[1]), log(lp.σf[1]), lp.β...]
    p = [AtAe, Atbe, AtAf, Atbf, α]
    prob = Optimization.OptimizationProblem(g, x0, p)
    sol = Optimization.solve(prob, OptimizationOptimisers.Adam(), maxiters = 1000)
    lp.σe .= exp(sol.u[1])
    lp.σf .= exp(sol.u[2])
    lp.β  .= sol.u[3:end]
    Q = pinv(
        Symmetric(
            lp.σe[1]^2 * pinv(Symmetric(AtAe), α) + lp.σf[1]^2 * pinv(Symmetric(AtAf), α),
        ),
    )
    lp.Σ .= Symmetric(Q)
end

"""
function learn!(
    lp::UnivariateLinearProblem,
    ss::SubsetSelector,
    α::Real;
    num_steps = 100,
    opt = Flux.Optimise.Adam()
)

Fit a univariate Gaussian distribution for the equation y = Aβ + ϵ, where β are model coefficients and ϵ ∼ N(0, σ). Fitting is done via batched gradient descent with batches provided by the subset selector and the gradients are calculated using Flux.  
"""
function learn!(
    lp::UnivariateLinearProblem,
    ss::SubsetSelector,
    α::Real;
    num_steps = 100,
    opt = Flux.Optimise.Adam()
)
    params = [log.(lp.σ); lp.β]
    f(x, p) = -logpdf(MvNormal(p[1] * x[2:end], exp(x[1])), p[2])
    for step = 1:num_steps
        inds = get_random_subset(ss)

        AtA = sum(v * v' for v in lp.iv_data[inds])
        Atb = sum(v * b for (v, b) in zip(lp.iv_data[inds], lp.dv_data[inds]))
        p = (AtA, Atb)
        if step % (num_steps ÷ 10) == 0
            err = @sprintf("%1.3e", f(params, p))
            println("Iteration #$(step): \t log(p(x)) = $err")
        end

        grads = Flux.gradient(x -> f(x, p), params)[1]
        Flux.Optimise.update!(opt, params, grads)
    end
    lp.σ .= exp(params[1])
    lp.β .= params[2:end]
    AtA = sum(v * v' for v in lp.iv_data)
    lp.Σ .= Symmetric(lp.σ[1]^2 * pinv(AtA, α))
end

"""
function learn!(
    lp::CovariateLinearProblem,
    ss::SubsetSelector,
    α::Real;
    num_steps=100,
    opt=Flux.Optimise.Adam()
)

Fit a Gaussian distribution by finding the MLE of the following log probability:
    ℓ(β, σe, σf) = -0.5*(e - A_e *β)'*(e - A_e * β) / σe - 0.5*(f - A_f *β)'*(f - A_f * β) / σf - log(σe) - log(σf)

through an iterative batch gradient descent optimization proceedure where the batches are provided by the subset selector. 
"""
function learn!(
    lp::CovariateLinearProblem,
    ss::SubsetSelector,
    α::Real;
    num_steps = 100,
    opt = Flux.Optimise.Adam()
)
    params = [log.(lp.σe); log.(lp.σf); lp.β]
    f(x, p) =
        -logpdf(MvNormal(p[1] * x[3:end], exp(x[1]) + p[5]), p[2]) -
        logpdf(MvNormal(p[3] * x[3:end], exp(x[2]) + p[5]), p[4])
    for step = 1:num_steps
        inds = get_random_subset(ss)

        AtAe = sum(b * b' for b in lp.B[inds])
        Atbe = sum(b * e for (b, e) in zip(lp.B[inds], lp.e[inds]))

        AtAf = sum(db * db' for db in lp.dB[inds])
        Atbf = sum(db * f for (db, f) in zip(lp.dB[inds], lp.f[inds]))

        p = (AtAe, Atbe, AtAf, Atbf) # TODO: should this have 5 parameters?
        if step % (num_steps ÷ 10) == 0
            err = @sprintf("%1.3e", f(params, p))
            println("Iteration #$(step): \t Batch log(p(x)) = $err")
        end

        grads = Flux.gradient(x -> f(x, p), params)
        Flux.Optimise.update!(opt, params, grads)
    end
    lp.σe .= exp(params[1])
    lp.σf .= exp(params[2])
    lp.β  .= params[3:end]
    AtAe = sum(b * b' for b in lp.B)
    AtAf = sum(db * db' for db in lp.dB)
    Q = pinv(
        Symmetric(
            lp.σe[1]^2 * pinv(Symmetric(AtAe), α) + lp.σf[1]^2 * pinv(Symmetric(AtAf), α),
        ),
    )
    lp.Σ .= Symmetric(Q)
end

# Weighted least squares functions #############################################

"""
function learn!(
    lp::UnivariateLinearProblem,
    ws::Vector,
    int::Bool
)

Fit energies using weighted least squares.
"""
function learn!(
    lp::UnivariateLinearProblem,
    ws::Vector,
    int::Bool
)
    @views B_train = reduce(hcat, lp.iv_data)'
    @views e_train = lp.dv_data

    # Calculate A and b.
    if int
        int_col = ones(size(B_train, 1))
        @views A = hcat(int_col, B_train)
    else
        @views A = B_train
    end
    @views b = e_train

    # Calculate coefficients β.
    Q = Diagonal(ws[1] * ones(length(e_train)))

    βs = Vector{Float64}() 
    try
        βs = (A'*Q*A) \ (A'*Q*b)
    catch e
        println(e)
        println("Linear system will be solved using pinv.") 
        βs = pinv(A'*Q*A)*(A'*Q*b)
    end
    
    # Update lp.
    if int
        lp.β0 .= βs[1]
        lp.β  .= βs[2:end]
    else
        lp.β  .= βs
    end
    
end

"""
function learn!(
    lp::CovariateLinearProblem,
    ws::Vector,
    int::Bool
)

Fit energies and forces using weighted least squares.
"""
function learn!(
    lp::CovariateLinearProblem,
    ws::Vector,
    int::Bool
)
    @views B_train = reduce(hcat, lp.B)'
    @views dB_train = reduce(hcat, lp.dB)'
    @views e_train = lp.e
    @views f_train = reduce(vcat, lp.f)
    
    # Calculate A and b.
    if int
        #int_col = ones(size(B_train, 1) + size(dB_train, 1))
        int_col = [ones(size(B_train, 1)); zeros(size(dB_train, 1))]
        @views A = hcat(int_col, [B_train; dB_train])
    else
        @views A = [B_train; dB_train]
    end
    @views b = [e_train; f_train]

    # Calculate coefficients βs.
    Q = Diagonal([ws[1] * ones(length(e_train));
                  ws[2] * ones(length(f_train))])
    
    βs = Vector{Float64}() 
    try
        βs = (A'*Q*A) \ (A'*Q*b)
    catch e
        println(e)
        println("Linear system will be solved using pinv.") 
        βs = pinv(A'*Q*A)*(A'*Q*b)
    end

    # Update lp.
    if int
        lp.β0 .= βs[1]
        lp.β  .= βs[2:end]
    else
        lp.β  .= βs
    end
    
end


"""
function learn!(
    lp::LinearProblem
)

Default learning problem: weighted least squares.
"""
function learn!(
    lp::LinearProblem
)
    n = typeof(lp) <: UnivariateLinearProblem ? 1 : 2
    ws, int = ones(n), false
    return learn!(lp, ws, int)
end

function assemble_matrices(lp, ws)
    @views B_train = reduce(hcat, lp.B)'
    @views dB_train = reduce(hcat, lp.dB)'
    @views e_train = lp.e
    @views f_train = reduce(vcat, lp.f)
    
    @views A = [B_train; dB_train]
    @views b = [e_train; f_train]

    W = Diagonal([ws[1] * ones(length(e_train));
                  ws[2] * ones(length(f_train))])

    A, W, b
end

function ooc_learn!(
    lb::InteratomicPotentials.LinearBasisPotential,
    ds_train::PotentialLearning.DataSet;
    ws = [30.0,1.0],
    symmetrize::Bool = true,
    lambda::Union{Real,Nothing} = 0.01,
    reg_style::Symbol = :default,
    AtWA = nothing,
    AtWb = nothing
)

    basis_size = length(lb.basis)

    if isnothing(AtWA) || isnothing(AtWb)
        AtWA = zeros(basis_size,basis_size)
        AtWb = zeros(basis_size)

        W = zeros(1,1)
   
        configs = get_system.(ds_train)
        
        for config in ds_train
            ref_energy = get_values(get_energy(config))
            ref_forces = reduce(vcat,get_values(get_forces(config)))

            sys = get_system(config)
            global_descrs = reshape(sum(compute_local_descriptors(sys,lb.basis)),:,1)'
            force_descrs  = stack(reduce(vcat,compute_force_descriptors(sys,lb.basis)))'

            A = [global_descrs; force_descrs]
            b = [ref_energy; ref_forces]
            if size(W)[1] != size(A)[1]
                W = Diagonal( [ws[1]*ones(length(ref_energy));
                               ws[2]*ones(length(ref_forces))])
            end

            AtWA .+= A'*W*A
            AtWb .+= A'*W*b
        end
    else
        AtWA = deepcopy(AtWA)
        AtWb = deepcopy(AtWb)
    end

    if symmetrize
        AtWA = Symmetric(AtWA)
    end
    
    if !isnothing(lambda)
        if reg_style == :default
            reg_matrix = lambda*Diagonal(ones(size(AtWA)[1]))
            AtWA += reg_matrix 
        end

        if reg_style == :scale_thresh || reg_style == :scale
            for i in 1:size(AtWA,1)
               reg_elem = AtWA[i,i]*(1+lambda)
               if reg_style == :scale_thresh
                   reg_elem = max(reg_elem,lambda)
               end#
               AtWA[i,i] = reg_elem        
            end
        end
            
    end

    β =  AtWA \ AtWb
    println("condition number of AtWA: $(cond(AtWA))")

    lb.β .= β   

    AtWA, AtWb
end
