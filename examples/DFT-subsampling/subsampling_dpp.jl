push!(Base.LOAD_PATH, "../../")

using PotentialLearning
using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using AtomsBase, Unitful, UnitfulAtomic
using InteratomicPotentials, InteratomicBasisPotentials
using Determinantal
using InvertedIndices
using CSV
using JLD
using DataFrames

include("subsampling_utils.jl")


# Load dataset -----------------------------------------------------------------
elname = "Hf" # "HfO"
elspec = [:Hf] # , :O]
inpath = "./DFT_data/$elname/"
outpath = "./DPP_training/$elname/"

# Read all data
file_arr = readext(inpath, "xyz")
nfile = length(file_arr)
confs_arr = [load_data(inpath*file, ExtXYZ(u"eV", u"Å")) for file in file_arr]

# Id of configurations per file
n = 0
confs_id = Vector{Vector{Int64}}(undef, nfile)
for k = 1:nfile
    confs_id[k] = (n+1):(n+length(confs_arr[k]))
    n += length(confs_arr[k])
end

# Read single file
# datafile = "Hf_mp100_EOS_1D_form_sorted.xyz"
# confs = load_data(inpath*datafile, ExtXYZ(u"eV", u"Å"))

# Define cases -----------------------------------------------------------------
param_sets = [[5,4], [5,6], [5,8]]
            #   [[5,4], [5,6], [5,8],
            #   [4,4], [4,6], [4,8]]
batch_sets = [10000] # , 5000, 1000, 500, 100, 50]

# Run experiments -----------------------------------------------------------------

for param in param_sets
    println("=========== PARAMS $param ===========")
    DPP_training_trial(elspec, param[1], param[2], batch_sets, file_arr, confs_arr, outpath; nfold=5)
end
    