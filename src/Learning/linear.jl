abstract type LinearProblem{T<:Real} <: AbstractLearningProblem end

struct UnivariateLinearProblem{T<:Real} <: LinearProblem{T}
    iv_data :: Vector 
    dv_data :: Vector 
    β       :: Vector{T}
    σ       :: Vector{T} 
end
struct CovariateLinearProblem{T<:Real} <: LinearProblem{T}
    e       :: Vector
    f       :: Vector{Vector}
    B       :: Vector{Vector}
    dB      :: Vector{Vector}
    β       :: Vector{T} 
    σe      :: Vector{T} 
    σf      :: Vector{T}
end

function LinearProblem(ds::DataSet{T}) where T 
    
    d_flag, descriptors, energies = try 
        true,  compute_features(ds, GlobalMean()), get_values.(get_energy.(ds))
    catch 
        false, 0.0 
    end
    fd_flag, force_descriptors, forces = try  
        true, get_force_descriptors.(ds), get_values.(get_forces.(ds))
    catch
        false, 0.0
    end
    
    if d_flag & ~fd_flag 
        dim = length(descriptors[1])
        β = zeros(T, dim)

        p = UnivariateLinearProblem(descriptors, 
                energies, 
                β, 
                [1.0])
    elseif ~d_flag & fd_flag 
        dim = length(force_descriptors[1][1])
        β = zeros(T, dim)

        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]
        p = UnivariateLinearProblem(force_descriptors,
            forces, 
            β, 
            [1.0]
        )
        
    elseif d_flag & fd_flag 
        dim_d = length(descriptors[1])
        dim_fd = length(force_descriptors[1][1])

        if  (dim_d != dim_fd) 
            error("Descriptors and Force Descriptors have different dimension!") 
        else
            dim = dim_d
        end

        β = zeros(T, dim)
        
        p = CovariateLinearProblem(energies, 
                forces, 
                descriptors, 
                force_descriptors, 
                β, 
                [1.0], 
                [1.0])

    else 
        error("Either no (Energy, Descriptors) or (Forces, Force Descriptors) in DataSet")
    end
    p
end

function learn!(lp::UnivariateLinearProblem; α = 1e-8)
    # Form design matrices 
    AtA = sum( v*v' for v in lp.iv_data)
    Atb = sum( v*b for (v,b) in zip(lp.iv_data, lp.dv_data))

    Q = pinv(AtA, α)
    copyto!(lp.β, Q*Atb) 
    copyto!(lp.σ, std(Atb - AtA*β))
    lp
end


function learn!(lp::CovariateLinearProblem; α = 1e-8)
    # Regularizaiton parameter α

    # Does not have analytical solution, use optimization 
    # break into energy and force components
    AtAe = sum( b*b' for b in lp.B)
    Atbe = sum( b*e for (b,e) in zip(lp.B, lp.e))

    AtAf = sum( db*db' for db in lp.dB)
    Atbf = sum( db*f for (db, f) in zip(lp.dB, lp.f))

    f(x, p) = -logpdf(MvNormal(p[1] * x[3:end], exp(x[1])+p[5]), p[2]) - logpdf(MvNormal(p[3] * x[3:end], exp(x[2])+p[5]), p[4])
    g = OptimizationFunction(f, Optimization.AutoForwardDiff())

    x0 = [lp.β..., log(lp.σe[1]), log(lp.σf[1])]
    p = [AtAe, Atbe, AtAf, Atbf, α]
    prob = Optimization.OptimizationProblem(g, x0, p)
    sol = Optimization.solve(prob, Optimization.BFGS())
    copyto!(lp.σe, exp(sol.u[1]))
    copyto!(lp.σf, exp(sol.u[2]))
    copyto!(lp.β, sol.u[3:end])
    lp 
end

function learn!(lp::UnivariateLinearProblem, ss::SubsetSelector; num_steps = 100, opt = Flux.Optimise.Adam())
    params = [lp.σ; lp.β]
    f(x, p) = -logpdf(MvNormal(p[1]*x[2:end], exp(x[1])), p[2])
    for step = 1:num_steps
        inds = get_random_subset(ss)

        AtA = sum( v*v' for v in lp.iv_data[inds])
        Atb = sum( v*b for (v,b) in zip(lp.iv_data[inds], lp.dv_data[inds]))
        p = (AtA, Atb)
        if step % (num_steps ÷ 10) == 0
            err = @sprintf("%1.3e", f(params, p))
            println("Iteration #$(step): \t log(p(x)) = $err")
        end
        
        grads = Flux.gradient(x->f(x, p), params)
        Flux.Optimise.update!(opt, params, grads)
    end
    copyto!(lp.σ, params[1])
    copyto!(lp.β, params[2:end])
    lp
end

function learn!(lp::CovariateLinearProblem, ss::SubsetSelector; num_steps = 100, opt = Flux.Optimise.Adam())
    params = [lp.σe; lp.σf; lp.β]
    f(x, p) = -logpdf(MvNormal(p[1] * x[3:end], exp(x[1])+p[5]), p[2]) - logpdf(MvNormal(p[3] * x[3:end], exp(x[2])+p[5]), p[4])
    for step = 1:num_steps
        inds = get_random_subset(ss)

        AtAe = sum( b*b' for b in lp.B[inds])
        Atbe = sum( b*e for (b,e) in zip(lp.B[inds], lp.e[inds]))

        AtAf = sum( db*db' for db in lp.dB[inds])
        Atbf = sum( db*f for (db, f) in zip(lp.dB[inds], lp.f[inds]))

        p = (AtAe, Atbe, AtAf, Atbf)
        if step % (num_steps ÷ 10) == 0
            err = @sprintf("%1.3e", f(params, p))
            println("Iteration #$(step): \t Batch log(p(x)) = $err")
        end
        
        grads = Flux.gradient(x->f(x, p), params)
        Flux.Optimise.update!(opt, params, grads)
    end
    copyto!(lp.σe, params[1])
    copyto!(lp.σf, params[2])
    copyto!(lp.β, params[3:end])
    lp
end


