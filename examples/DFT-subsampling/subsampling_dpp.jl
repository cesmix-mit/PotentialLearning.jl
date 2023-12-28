push!(Base.LOAD_PATH, "../../")

using PotentialLearning
using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using AtomsBase, Unitful, UnitfulAtomic
using InteratomicPotentials
using Determinantal
using CairoMakie
using InvertedIndices
using CSV
using JLD
using DataFrames

include("subsampling_utils.jl")

# Load dataset -----------------------------------------------------------------
elname = "Si"
elspec = [:Si]
inpath = "../Si-3Body-LAMMPS/"
outpath = "./output/$elname/"

# Read all data
file_arr = readext(inpath, "xyz")
nfile = length(file_arr)
confs_arr = [load_data(inpath*file, ExtXYZ(u"eV", u"Å")) for file in file_arr]
confs = concat_dataset(confs_arr)

# Id of configurations per file
n = 0
confs_id = Vector{Vector{Int64}}(undef, nfile)
for k = 1:nfile
    global n
    confs_id[k] = (n+1):(n+length(confs_arr[k]))
    n += length(confs_arr[k])
end

# Read single file
# datafile = "Hf_mp100_EOS_1D_form_sorted.xyz"
# confs = load_data(inpath*datafile, ExtXYZ(u"eV", u"Å"))

# Define ACE basis -------------------------------------------------------------
nbody = 4
deg = 5
ace = ACE(species = elspec,             # species
          body_order = nbody,           # n-body
          polynomial_degree = deg,      # degree of polynomials
          wL = 1.0,                     # Defaults, See ACE.jl documentation 
          csp = 1.0,                    # Defaults, See ACE.jl documentation 
          r0 = 1.0,                     # minimum distance between atoms
          rcutoff = 10.0)

# Update dataset by adding energy (local) descriptors --------------------------
println("Computing local descriptors")
@time e_descr = compute_local_descriptors(confs, ace)
@time f_descr = compute_force_descriptors(confs, ace)
JLD.save(outpath*"$(elname)_energy_descriptors.jld", "e_descr", e_descr)
JLD.save(outpath*"$(elname)_force_descriptors.jld", "f_descr", f_descr)

ds = DataSet(confs .+ e_descr .+ f_descr)
ndata = length(ds)

# Compute cross validation error from training ---------------------------------
batch_size = [80, 40, 20]
sel_ind = Dict{Int64, Vector}()
cond_num = Dict{Int64, Vector}()

for bs in batch_size
    println("=============== Starting batch size $bs ===============")
    sel_ind[bs], cond_num[bs] = cross_validation_training(ds; ndiv=5, dpp_batch=bs)
end

JLD.save(outpath*"$(elname)_ACE-$(nbody)-$(deg)_DPP_indices_and_condnum.jld",
    "ind", sel_ind,
    "condnum", cond_num)

