"""
    LearningProblem{T<:Real} <: AbstractLearningProblem 
        ds      :: DataSet
        logprob :: Function # logprob(y::params, ds::Dataset) -> scalar
        ∇logprob :: Function # ∇logprob(y::params, ds::Dataset) -> gradient wrt y (params)
        params :: Vector{T} # model parameters
    
Generic LearningProblem that allows the user to pass a logprob(y::params, ds::DataSet) function and its gradient. The gradient should return a vector of logprob with respect to it's params. If the user does not have a gradient function available, then Flux can provide one for it (provided that logprob is of the form above).
"""
struct LearningProblem{T<:Real} <: AbstractLearningProblem 
    ds      :: DataSet
    logprob :: Function 
    ∇logprob :: Function 
    params :: Vector{T}
end

function LearningProblem(ds :: DataSet, logprob :: Function, params :: Vector{T}) where T
    ∇logprob(x, ds) = Flux.gradient(y->logprob(y, ds), x)
    LearningProblem(ds, logprob, ∇logprob, params)
end

"""
    learn!(lp::LearningProblem;; num_steps = 100 :: Int, opt = Flux.Optimisers.Adam()) 

Attempts to fit the parameters lp.params in the learning problem lp using gradient descent with the optimizer opt and num_steps number of iterations.
"""
function learn!(lp::LearningProblem; num_steps = 100 :: Int, opt = Flux.Optimisers.Adam()) 
    for step = 1:num_steps 
        #if step % (num_steps ÷ 10) == 0
            err = @sprintf("%1.3e", lp.logprob(lp.params, lp.ds))
            println("Iteration #$(step): \t Batch log(p(x)) = ", lp.logprob(lp.params, lp.ds))
        #end
        grads = lp.∇logprob(lp.params, lp.ds)
        Flux.Optimise.update!(opt, lp.params, grads)
    end
end
"""
    learn!(lp::LearningProblem, ss::SubsetSelector; num_steps = 100 :: Int, opt = Flux.Optimisers.Adam())  

Attempts to fit the parameters lp.params in the learning problem lp using batch gradient descent with the optimizer opt and num_steps number of iterations. Batching is provided by the passed ss::SubsetSelector. 
"""
function learn!(lp::LearningProblem, ss::SubsetSelector; num_steps = 100 :: Int, opt = Flux.Optimisers.Adam()) 
    for step = 1:num_steps 
        inds = get_random_subset(ss)
        if step % (num_steps ÷ 10) == 0
            err = @sprintf("%1.3e", lp.logprob(lp.params, lp.ds[inds]))
            println("Iteration #$(step): \t Batch log(p(x)) = ", lp.logprob(lp.params, lp.ds[inds]))
        end
        grads = lp.∇logprob(lp.params, lp.ds[inds])
        Flux.Optimise.update!(opt, lp.params, grads)
    end
end
