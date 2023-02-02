###########################################################
###########################################################
## Featurization 
"""
    Feature

A struct of abstract type Feature represents a function that takes in a set of local descriptors corresponding to some atomic environment and produce a `global` descriptor. 
"""
abstract type Feature end
"""
        GlobalSum{T}

GlobalSum produces the sum of the local descriptors.
"""
struct GlobalSum <: Feature end
GlobalSum(nothing) = GlobalSum


function compute_feature(B::LocalDescriptors, gm::GlobalSum) where T
    sum(get_values(B))
end 

"""
        GlobalMean{T}

GlobalMean produces the mean of the local descriptors.
"""
struct GlobalMean <: Feature end
GlobalMean(nothing) = GlobalMean


function compute_feature(B::LocalDescriptors, gm::GlobalMean) where T
    mean(get_values(B))
end 

#####################################################
#####################################################

"""
    CorrelationMatrix 
        α :: Vector{Float64} # weights

CorrelationMatrix produces a global descriptor that is the correlation matrix of the local descriptors. In other words, it is mean(bi'*bi for bi in B). 
"""
struct CorrelationMatrix <: Feature end
CorrelationMatrix(nothing) = CorrelationMatrix

function compute_feature(B::LocalDescriptors, cm::CorrelationMatrix) 
    Symmetric(mean(Bi*Bi' for Bi in get_values(B)))
end

function compute_feature(dB::ForceDescriptors, cm::CorrelationMatrix) 
    ## Remember that dB_i is a 3 vector of descriptors of length dim 
    CorrelationMatrix(   Symmetric(   mean(reduce(hcat, dBi)*reduce(hcat,dBi)' for dBi in get_values(dB))   )   )
end

function compute_feature(c::Configuration, gm::GlobalMean; dt = LocalDescriptors)
    compute_feature(get_local_descriptors(c), gm)
end

function compute_feature(c::Configuration, gs::GlobalSum; dt = LocalDescriptors)
    compute_feature(get_local_descriptors(c), gs)
end

function compute_feature(c::Configuration, cm::CorrelationMatrix; dt = LocalDescriptors)
    compute_feature(c.data[dt], cm)
end

"""
    compute_feature(ds::DataSet, f::Feature; dt = LocalDescriptors)

Computes features of the dataset ds using the feature method F on descriptors dt (default option are the LocalDescriptors, if available).
"""
function compute_features(ds::DataSet, f::Feature; dt = LocalDescriptors)
    compute_feature.(ds, (f,); dt = dt)
end
