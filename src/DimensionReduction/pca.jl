"""
    struct PCA <: DimensionReducer
        tol :: Float64 
    end

Use SVD to compute the PCA of the design matrix of descriptors. (using Force descriptors TBA)

If tol is a float then the number of components to keep is determined by the smallest n such that relative percentage of variance explained by keeping the leading n principle components is greater than 1 - tol. If tol is an int, then we return the components corresponding to the tol largest eigenvalues.
"""
struct PCA{T <: Real} <: DimensionReducer
    tol :: T
end

function PCA(; tol = 0.01)
    PCA(tol)
end

"""
    fit(ds::DataSet, pca::PCA)

Fits a linear dimension reduction routine using PCA on the global descriptors in the dataset ds. 
"""
function fit(ds::DataSet, pca :: PCA)
    d = try 
        get_values.(get_descriptors.(ds))
    catch
        error("No local descriptors found in DataSet")
    end
    λ, W = select_eigendirections(d, pca.tol)
end
