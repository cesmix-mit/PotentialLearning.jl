"""
    LinearPRoblem{T<:Real} <: AbstractLearningProblem 

An abstract type to specify linear potential inference problems. 
"""
abstract type LinearProblem{T<:Real} <: AbstractLearningProblem end
"""
    UnivariateLinearProblem{T<:Real} <: LinearProblem{T}
        iv_data :: Vector 
        dv_data :: Vector 
        β       :: Vector{T}
        σ       :: Vector{T} 
        Σ       :: Symmetric{T, Matrix{T}}
    end

A UnivariateLinearProblem is a linear problem in which there is only 1 type of independent variable / dependent variable. Typically, that means we are either only fitting energies or only fitting forces. When this is the case, the solution is available analytically and the standard deviation, σ, and covariance, Σ, of the coefficients, β, are computable. 
"""
struct UnivariateLinearProblem{T<:Real} <: LinearProblem{T}
    iv_data::Vector
    dv_data::Vector
    β::Vector{T}
    σ::Vector{T}
    Σ::Symmetric{T,Matrix{T}}
end
Base.show(io::IO, u::UnivariateLinearProblem{T}) where {T} =
    print(io, "UnivariateLinearProblem{T, $(u.β), $(u.σ)}")
"""
    CovariateLinearProblem{T<:Real} <: LinearProblem{T}
        e       :: Vector
        f       :: Vector{Vector{T}}
        B       :: Vector{Vector{T}}
        dB      :: Vector{Matrix{T}}
        β       :: Vector{T} 
        σe      :: Vector{T} 
        σf      :: Vector{T}
        Σ       :: Symmetric{T, Matrix{T}}
    

A CovariateLinearProblem is a linear problem in which we are fitting energies and forces using both descriptors and their gradients (B and dB, respectively). When this is the case, the solution is not available analytically and must be solved using some iterative optimization proceedure. In the end, we fit the model coefficients, β, standard deviations corresponding to energies and forces, σe and σf, and the covariance Σ. 
"""
struct CovariateLinearProblem{T<:Real} <: LinearProblem{T}
    e::Vector
    f::Vector{Vector{T}}
    B::Vector{Vector{T}}
    dB::Vector{Matrix{T}}
    β::Vector{T}
    σe::Vector{T}
    σf::Vector{T}
    Σ::Symmetric{T,Matrix{T}}
end

Base.show(io::IO, u::CovariateLinearProblem{T}) where {T} =
    print(io, "CovariateLinearProblem{T, $(u.β), $(u.σe), $(u.σf)}")
"""
    LinearProblem(ds::DatasSet; T = Float64)

Construct a LinearProblem by detecting if there are energy descriptors and/or force descriptors and construct the appropriate LinearProblem (either Univariate, if only a single type of descriptor, or Covariate, if there are both types).
"""
function LinearProblem(ds::DataSet; T = Float64)

    d_flag, descriptors, energies = try
        true, compute_features(ds, GlobalSum()), get_values.(get_energy.(ds))
    catch
        false, 0.0, 0.0
    end
    fd_flag, force_descriptors, forces = try
        true,
        [reduce(vcat, get_values(get_force_descriptors(dsi))) for dsi in ds],
        get_values.(get_forces.(ds))
    catch
        false, 0.0, 0.0
    end
    if d_flag & ~fd_flag
        dim = length(descriptors[1])
        β = zeros(T, dim)

        p = UnivariateLinearProblem(
            descriptors,
            energies,
            β,
            [1.0],
            Symmetric(zeros(dim, dim)),
        )
    elseif ~d_flag & fd_flag
        dim = length(force_descriptors[1][1])
        β = zeros(T, dim)

        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]
        p = UnivariateLinearProblem(
            force_descriptors,
            [reduce(vcat, fi) for fi in forces],
            β,
            [1.0],
            Symmetric(zeros(dim, dim)),
        )

    elseif d_flag & fd_flag
        dim_d = length(descriptors[1])
        dim_fd = length(force_descriptors[1][1])

        if (dim_d != dim_fd)
            error("Descriptors and Force Descriptors have different dimension!")
        else
            dim = dim_d
        end

        β = zeros(T, dim)
        forces = [reduce(vcat, fi) for fi in forces]
        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]

        p = CovariateLinearProblem(
            energies,
            [reduce(vcat, fi) for fi in forces],
            descriptors,
            force_descriptors,
            β,
            [1.0],
            [1.0],
            Symmetric(zeros(dim, dim)),
        )

    else
        error("Either no (Energy, Descriptors) or (Forces, Force Descriptors) in DataSet")
    end
    p
end
"""
    learn!(lp::UnivariateLinearProblem; α = 1e-8)

Fit a univariate Gaussian distribution for the equation y = Aβ + ϵ, where β are model coefficients and ϵ ∼ N(0, σ). Fitting is done via SVD on the design matrix, A'*A (formed iteratively), where eigenvalues less than α are cut-off.  
"""
function learn!(lp::UnivariateLinearProblem; α = 1e-8)
    # Form design matrices 
    AtA = sum(v * v' for v in lp.iv_data)
    Atb = sum(v * b for (v, b) in zip(lp.iv_data, lp.dv_data))

    Q = pinv(AtA, α)
    copyto!(lp.β, Q * Atb)
    copyto!(lp.σ, std(Atb - AtA * lp.β))
    copyto!(lp.Σ, Symmetric(lp.σ[1]^2 * Q))
    lp
end

"""
    learn!(lp::CovariateLinearProblem; α = 1e-8)

Fit a Gaussian distribution by finding the MLE of the following log probability:
    ℓ(β, σe, σf) = -0.5*(e - A_e *β)'*(e - A_e * β) / σe - 0.5*(f - A_f *β)'*(f - A_f * β) / σf - log(σe) - log(σf)

through an optimization proceedure. 
"""
function learn!(lp::CovariateLinearProblem; α = 1e-8)
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

    x0 = [lp.β..., log(lp.σe[1]), log(lp.σf[1])]
    p = [AtAe, Atbe, AtAf, Atbf, α]
    prob = Optimization.OptimizationProblem(g, x0, p)
    sol = Optimization.solve(prob, Optim.BFGS())
    copyto!(lp.σe, exp(sol.u[1]))
    copyto!(lp.σf, exp(sol.u[2]))
    copyto!(lp.β, sol.u[3:end])
    Q = pinv(
        Symmetric(
            lp.σe[1]^2 * pinv(Symmetric(AtAe), α) + lp.σf[1]^2 * pinv(Symmetric(AtAf), α),
        ),
    )
    copyto!(lp.Σ, Symmetric(Q))
    lp
end
"""
    learn!(lp::UnivariateLinearProblem, ss::SubsetSelector; α = 1e-8)

Fit a univariate Gaussian distribution for the equation y = Aβ + ϵ, where β are model coefficients and ϵ ∼ N(0, σ). Fitting is done via batched gradient descent with batches provided by the subset selector and the gradients are calculated using Flux.  
"""
function learn!(
    lp::UnivariateLinearProblem,
    ss::SubsetSelector;
    num_steps = 100,
    opt = Flux.Optimise.Adam(),
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
    copyto!(lp.σ, exp(params[1]))
    copyto!(lp.β, params[2:end])
    AtA = sum(v * v' for v in lp.iv_data)
    copyto!(lp.Σ, Symmetric(lp.σ[1]^2 * pinv(AtA, α)))

    lp
end
"""
    learn!(lp::CovariateLinearProblem, ss::SubsetSelector; α = 1e-8)

Fit a Gaussian distribution by finding the MLE of the following log probability:
    ℓ(β, σe, σf) = -0.5*(e - A_e *β)'*(e - A_e * β) / σe - 0.5*(f - A_f *β)'*(f - A_f * β) / σf - log(σe) - log(σf)

through an iterative batch gradient descent optimization proceedure where the batches are provided by the subset selector. 
"""
function learn!(
    lp::CovariateLinearProblem,
    ss::SubsetSelector;
    num_steps = 100,
    opt = Flux.Optimise.Adam(),
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

        p = (AtAe, Atbe, AtAf, Atbf)
        if step % (num_steps ÷ 10) == 0
            err = @sprintf("%1.3e", f(params, p))
            println("Iteration #$(step): \t Batch log(p(x)) = $err")
        end

        grads = Flux.gradient(x -> f(x, p), params)
        Flux.Optimise.update!(opt, params, grads)
    end
    copyto!(lp.σe, exp(params[1]))
    copyto!(lp.σf, exp(params[2]))
    copyto!(lp.β, params[3:end])
    AtAe = sum(b * b' for b in lp.B)
    AtAf = sum(db * db' for db in lp.dB)
    Q = pinv(
        Symmetric(
            lp.σe[1]^2 * pinv(Symmetric(AtAe), α) + lp.σf[1]^2 * pinv(Symmetric(AtAf), α),
        ),
    )
    copyto!(lp.Σ, Symmetric(Q))

    lp
end
"""
    learn!(lb::LBasisPotential, ds::DataSet; α = 1e-8, return_cov = true)

Fits the linear basis potential lb using data from DataSet using either an analytic solution or an optimization proceedure, depending on whether both energies and forces are present. Optionally returns the posterior covariance over the parameters. 
"""
function learn!(
    lb::InteratomicPotentials.LinearBasisPotential,
    ds::DataSet;
    α = 1e-8,
    return_cov = true,
)
    linear_problem = LinearProblem(ds)
    learn!(linear_problem; α = α)
    copy!(lb.β, linear_problem.β)
    if return_cov
        lb, linear_problem.Σ
    else
        lb
    end
end
"""
    learn!(lb::LBasisPotential, ds::DataSet, ss::SubsetSelector; α = 1e-8, return_cov = true)

Fits the linear basis potential lb using data from DataSet using batch gradient descent with batches provided by the subset selector ss. Optionally returns the posterior covariance over the parameters. 
"""
function learn!(
    lb::InteratomicPotentials.LinearBasisPotential,
    ds::DataSet,
    ss::SubsetSelector;
    α = 1e-8,
    return_cov = true,
)
    linear_problem = LinearProblem(ds)
    learn!(linear_problem, ss; α = α)
    copy!(lb.β, linear_problem.β)
    if return_cov
        lb, linear_problem.Σ
    else
        lb
    end
end
