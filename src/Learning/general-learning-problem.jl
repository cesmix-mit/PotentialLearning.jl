"""
struct LearningProblem{T<:Real} <: AbstractLearningProblem
    ds::DataSet
    logprob::Function
    ∇logprob::Function
    params::Vector{T}
end
 
Generic LearningProblem that allows the user to pass a logprob(y::params, ds::DataSet) function and its gradient. The gradient should return a vector of logprob with respect to it's params. If the user does not have a gradient function available, then Flux can provide one for it (provided that logprob is of the form above).
"""
struct LearningProblem{T<:Real} <: AbstractLearningProblem
    ds::DataSet
    logprob::Function
    ∇logprob::Function
    params::Vector{T}
end

"""
function LearningProblem(
    ds::DataSet,
    logprob::Function,
    params::Vector{T}
) where {T}

Generic LearningProblem construnctor.
"""
function LearningProblem(
    ds::DataSet,
    logprob::Function,
    params::Vector{T}
) where {T}
    ∇logprob(x, ds) = Flux.gradient(y -> logprob(y, ds), x)
    return LearningProblem(ds, logprob, ∇logprob, params)
end