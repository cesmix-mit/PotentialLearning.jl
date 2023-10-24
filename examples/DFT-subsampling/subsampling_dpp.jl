push!(Base.LOAD_PATH, "../../")

using PotentialLearning
using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using AtomsBase, Unitful, UnitfulAtomic
using InteratomicPotentials, InteratomicBasisPotentials
using Determinantal
using CairoMakie
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
confs = concat_dataset(confs_arr)

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
param_sets = [[4,4], [4,6], [4,8],
                     [5,6], [5,8],
                            [6,8],
                            [7,8]]
batch_sets = [5000, 2000, 1000, 500, 200, 100, 50]

# Run experiments -----------------------------------------------------------------

for param in param_sets
    @async DPP_training_trial(elspec, confs, param[1], param[2], batch_sets, outpath; nfold=5)
end
    