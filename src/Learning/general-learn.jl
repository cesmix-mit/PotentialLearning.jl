"""
function learn!(
    lp::LearningProblem;
    num_steps=100::Int,
    opt=Flux.Optimisers.Adam()
)

Attempts to fit the parameters lp.params in the learning problem lp using gradient descent with the optimizer opt and num_steps number of iterations.
"""
function learn!(
    lp::LearningProblem;
    num_steps = 100::Int,
    opt = Flux.Optimisers.Adam()
)
    for step = 1:num_steps
        if step % (num_steps ÷ 10) == 0
            err = @sprintf("%1.3e", lp.logprob(lp.params, lp.ds))
            println("Iteration #$(step): \t log(p(x)) = $err")
        end
        grads = lp.∇logprob(lp.params, lp.ds)
        Flux.Optimise.update!(opt, lp.params, grads)
    end
end

"""
function learn!(
    lp::LearningProblem,
    ss::SubsetSelector;
    num_steps = 100::Int,
    opt = Flux.Optimisers.Adam()
)

Attempts to fit the parameters lp.params in the learning problem lp using batch gradient descent with the optimizer opt and num_steps number of iterations. Batching is provided by the passed ss::SubsetSelector. 
"""
function learn!(
    lp::LearningProblem,
    ss::SubsetSelector;
    num_steps = 100::Int,
    opt = Flux.Optimisers.Adam()
)
    for step = 1:num_steps
        inds = get_random_subset(ss)
        if step % (num_steps ÷ 10) == 0
            err = @sprintf("%1.3e", lp.logprob(lp.params, lp.ds[inds]))
            println("Iteration #$(step): \t Batch log(p(x)) = $err")
        end
        grads = lp.∇logprob(lp.params, lp.ds[inds])
        Flux.Optimise.update!(opt, lp.params, grads)
    end
end
