module PotentialLearning

using AtomsBase
using InteratomicPotentials
using Unitful, UnitfulAtomic
using LinearAlgebra, Statistics, Random, Distributions
using StaticArrays
using OrderedCollections
using Flux
using Zygote
using Optimization
using OptimizationOptimisers
using Printf

# Custom Adjoints for StaticVectors
Zygote.@adjoint (T::Type{<:SVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)

## Data structs 
include("Data/data.jl")

# Kernel structs and functions 
include("Kernels/kernels.jl")

# Data input/output
include("io/io.jl")

# Subset selection
include("SubsetSelection/subsetselector.jl")

# Dimension Reduction
include("DimensionReduction/dimension_reduction.jl")

# Learning problems 
include("Learning/learning.jl")

# Metrics 
include("Metrics/metrics.jl") 

end
