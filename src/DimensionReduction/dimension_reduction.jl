abstract type DimensionReducer end 
export DimensionReducer, PCA, ActiveSubspace, fit, fit_transform, select_eigendirections
"""
    fit(ds::DataSet, dr::DimensionReducer)

Fits a linear dimension reduction routine using information from DataSet. See individual types of DimensionReducers for specific details.
"""
function fit end

function compute_eigen(d::Vector{T}) where T <: Vector{<:Real}
    Q = Symmetric(mean(di*di' for di in d))
    eigen(Symmetric(Q))
end
function select_eigendirections(d::Vector{T}, tol :: Float64 ) where T <: Vector{<:Real}
    λ, ϕ = compute_eigen(d)
    λ, ϕ = λ[end:-1:1], ϕ[end:-1:1, :] # reorder
    Σ = 1.0 .- cumsum(λ) / sum(λ)
    W = ϕ[Σ .> tol, :]
    λ, W
end
function select_eigendirections(d::Vector{T}, tol :: Int ) where T <: Vector{<:Real}
    λ, ϕ = compute_eigen(d)
    λ, ϕ = λ[end:-1:1], ϕ[end:-1:1, :] # reorder
    Σ = 1.0 .- cumsum(λ) / sum(λ)
    W = ϕ[1:tol, :]
    λ, W
end

include("pca.jl")
include("as.jl")
"""
    fit_transform(ds::DataSet, dr::DimensionReducer)

Fits a linear dimension reduction routine using information from DataSet and performs dimension reduction on descriptors and force_descriptors (whichever are available). See individual types of DimensionReducers for specific details.
"""
function fit_transform(ds::DataSet, dr :: DimensionReducer)
    W = fit(ds, dr)

    ds̃ = try 
        l = get_descriptors.(ds)
        l = (W, ) .* l
        ds .+ l
    catch
        ds
    end
    ds̃ = try 
        fd = get_force_descriptors.(ds)
        fd = (W, ) .* fd 
        ds .+ fd 
    catch
        ds̃
    end

    ds̃
end

