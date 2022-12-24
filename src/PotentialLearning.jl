module PotentialLearning

using LinearAlgebra, Statistics, Random, Distributions
using Unitful, UnitfulAtomic, AtomsBase
using StaticArrays
using Zygote 
using InteratomicPotentials
using OrderedCollections
using IterTools
using CSV
using Plots

# Custom Adjoints for StaticVectors
@Zygote.adjoint (T::Type{<:SVector})(x::AbstractVector) = T(x), dv -> (nothing, dv)

## Data structs 
include("Data/data.jl")

# Kernel structs and functions 
include("Kernels/kernels.jl")

# Data input/output
include("IO/io.jl")

# Subset selection
include("SubsetSelection/subsetselector.jl")

# Dimension Reduction
include("DimensionReduction/dimension_reduction.jl")

# Learning problems 
include("Learning/learning.jl")

# Metrics 
include("Metrics/metrics.jl") 

# Postproc
include("PostProc/Plots.jl")

end
