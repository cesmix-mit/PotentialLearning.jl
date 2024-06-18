"""
    PCAState <: DimensionReducer
        tol :: Float64 

Use SVD to compute the PCA of the design matrix of descriptors.

If tol is a float then the number of components to keep is determined by the smallest n such that relative percentage of variance explained by keeping the leading n principle components is greater than 1 - tol. If tol is an int, then we return the components corresponding to the tol largest eigenvalues.
"""
mutable struct PCAState <: DimensionReducer
    tol
    λ
    W
    m
end

function PCAState(; tol = 0.01, λ = [], W = [], m = [])
    PCAState(tol, λ, W, m)
end

function fit!(ds::DataSet, pca::PCAState)
    d = try
        #vcat(get_values.(get_local_descriptors.(ds))...) # use local desc
        sum.(get_values.(get_local_descriptors.(ds))) # use global desc
    catch
        error("No local descriptors found in DataSet")
    end
    d = try
        f = get_values.(get_force_descriptors.(ds))
        ff = vcat(vcat(fd...)...)
        return vcat(d, ff)
    catch
        d
    end
    if pca.m == []
        pca.m = sum(d) / length(d)
    end
    dm = d .- [pca.m] # center desc
    pca.λ, pca.W = select_eigendirections(dm, pca.tol)
    nothing
end

function transform!(ds::DataSet, dr::DimensionReducer)
    ds̃ = try
        ldc = get_values.(get_local_descriptors.(ds))
        ml = dr.m / length(ldc[1]) # compute local mean
        ldc_new = [LocalDescriptors([(dr.W' * (l .- ml)) for l in ld])
                   for ld in ldc]
        ds .+ ldc_new
    catch
        ds
    end
    ds̃ = try
        fdc = get_values.(get_force_descriptors.(ds))
        fdc_new = [ForceDescriptors([[(dr.W' * (fc .- dr.m)) for fc in f] for f in fd])
                   for fd in fdc]
        ds̃ .+ fdc_new
    catch
        ds̃
    end
    ds̃ = DataSet(ds̃)
    copyto!(ds.Configurations, ds̃.Configurations)
end


