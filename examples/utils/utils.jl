using Statistics
using OrderedCollections
using StaticArrays
using IterTools
using ProgressBars
using Plots

include("input.jl")
include("macros.jl")
include("NNIAP.jl")
include("plots.jl")


# Missing function
Base.size(fd::ForceDescriptors) = (length(fd), )





