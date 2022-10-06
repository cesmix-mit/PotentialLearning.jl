"""
    struct ActiveSubspace{T<:Real} <: DimensionReducer
        Q :: Function 
        ∇Q :: Function (gradient of Q)
        tol :: T
    end

Use the theory of active subspaces, with a given quantity of interest (expressed as the function Q) which takes a Configuration as an input and outputs a real scalar. ∇Q should input a Configuration and output an appropriate gradient. 
If tol is a float then the number of components to keep is determined by the smallest n such that relative percentage of variance explained by keeping the leading n principle components is greater than 1 - tol. If tol is an int, then we return the components corresponding to the tol largest eigenvalues.
"""
struct ActiveSubspace{T<:Real} <: DimensionReducer
    Q :: Function 
    ∇Q :: Function 
    tol :: T 
end

function ActiveSubspace(Q::Function, ∇Q::Function; tol = 0.01)
    ActiveSubspace(Q, ∇Q, tol)
end
"""
    fit(ds::DataSet, as::ActiveSubspace)

Fits a linear dimension reduction routine using the eigendirections of the uncentered covariance of the function ∇Q(c::Configuration) over the configurations in ds. Primarily used to reduce the dimension of the descriptors.
"""
function fit(ds::DataSet, as :: ActiveSubspace)
    d = [as.∇Q(c) for c in ds]
    λ, W = select_eigendirections(d, as.tol)
end



