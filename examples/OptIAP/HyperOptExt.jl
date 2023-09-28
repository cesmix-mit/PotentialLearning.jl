# Extension of HyperOpt.jl to handle the following cases:
#    - Allow `results` in `Hyperoptimizer` contain other elements: `loss` and `opt_iap`.
#    - A temporary common interface for all samplers based on the function `inject_pars`.


import Base: <, isless, ==, isequal, isinf
export <, isless, ==, isequal, isinf

struct HOResult <: Number
    loss
    opt_iap
    time
end

<(x::HOResult, y::HOResult) = x.loss < y.loss
<(x::Number, y::HOResult) = x < y.loss
<(x::HOResult, y::Number) = x.loss < y
isless(x::HOResult, y::HOResult) = x.loss < y.loss
isless(x::Number, y::HOResult) = x < y.loss
isless(x::HOResult, y::Number) = x.loss < y
==(x::HOResult, y::HOResult) = x.loss == y.loss
==(x::Number, y::HOResult) = x == y.loss
==(x::HOResult, y::Number) = x.loss == y
isequal(x::HOResult, y::HOResult) = x.loss == y.loss
isequal(x::Number, y::HOResult) = x == y.loss
isequal(x::HOResult, y::Number) = x.loss == y
Real(x::HOResult) = x.loss
Float64(x::HOResult) = Float64(x.loss)
isinf(x::HOResult) = isinf(x.loss)


function inject_pars(ho_pars, model_pars, ex)
    # Inject hyper-optimizer parameteres
    e = ex.args[2].args[3]
    e.args[1].args = []
    for (k,v) in eval(ho_pars)
        push!(e.args[1].args, :($k = $v))
    end
    
    # Inject model parameteres
    for (k,v) in eval(model_pars)
        push!(e.args[1].args, :($k = $v))
    end
    
    # Inject state variable
    state = [k for k in keys(eval(model_pars))]
    state_cond = Meta.parse(string("if state == nothing state = [", ["$k," for k in state]..., "] end"))
    pushfirst!(e.args[2].args, state_cond)
    
    # Inject state variable in return tuple, only when Hyperband is used
    if typeof(eval(ho_pars)[:sampler]) == Hyperband
        ret_e = e.args[2].args[end-1]
        e.args[2].args[end-1] = Meta.parse(string("$ret_e, state"))
    end

    # Evaluate new code
    eval(ex)
end

function print(hyper_optimizer)
    losses = map(x -> x.loss, hyper_optimizer.results)
    sortslices(hcat(losses, hyper_optimizer.history),
               dims = 1, by = x->x[1])
end
