# # Subsample Si dataset and fit with ACE

# ## Load packages, define paths, and create experiment folder.

# Load packages.
using LinearAlgebra, Random, InvertedIndices
using Statistics, StatsBase, Distributions, Determinantal
using Unitful, UnitfulAtomic
using AtomsBase, InteratomicPotentials, PotentialLearning
using CSV, JLD, DataFrames

# Define atomic type information.
elname, elspec = "Si", [:Si] 

# Define paths.
path = joinpath(dirname(pathof(PotentialLearning)), "../examples/DPP-ACE-Si")
inpath = "$path/../data/Si-3Body-LAMMPS/"
outpath = "$path/output/$elname/"

# Load utility functions.
include("$path/subsampling_utils.jl")

# ## Load atomistic datasets.

# Load all atomistic datasets: atomistic configurations (atom positions, geometry, etc.) + DFT data (energies, forces, etc.)
file_arr = readext(inpath, "xyz")
nfile = length(file_arr)
confs_arr = [load_data(inpath*file, ExtXYZ(u"eV", u"Å")) for file in file_arr]
confs = concat_dataset(confs_arr)

# Id of configurations per file.
n = 0
confs_id = Vector{Vector{Int64}}(undef, nfile)
for k = 1:nfile
    global n
    confs_id[k] = (n+1):(n+length(confs_arr[k]))
    n += length(confs_arr[k])
end

# ## Subsampling by DPP.

# Create ACE basis.
nbody = 4
deg = 5
ace = ACE(species = elspec,             # species
          body_order = nbody,           # n-body
          polynomial_degree = deg,      # degree of polynomials
          wL = 1.0,                     # Defaults, See ACE.jl documentation 
          csp = 1.0,                    # Defaults, See ACE.jl documentation 
          r0 = 1.0,                     # minimum distance between atoms
          rcutoff = 10.0)

# Compute ACE descriptors for energies and forces.
println("Computing local descriptors")
e_descr = compute_local_descriptors(confs, ace; pbar=false)
f_descr = compute_force_descriptors(confs, ace; pbar=false)
JLD.save(outpath*"$(elname)_energy_descriptors.jld", "e_descr", e_descr)
JLD.save(outpath*"$(elname)_force_descriptors.jld", "f_descr", f_descr)

# Update training dataset by adding energy and force descriptors.
ds = DataSet(confs .+ e_descr .+ f_descr)
ndata = length(ds)

# ## Compute cross validation error from training dataset.
batch_size = [80, 40]
sel_ind = Dict{Int64, Vector}()
cond_num = Dict{Int64, Vector}()

for bs in batch_size
    println("=============== Starting batch size $bs ===============")
    sel_ind[bs], cond_num[bs] = cross_validation_training(ds; ndiv=5, dpp_batch=bs)
    println("condnum: $(cond_num[bs])")
end

JLD.save(outpath*"$(elname)_ACE-$(nbody)-$(deg)_DPP_indices_and_condnum.jld",
    "ind", sel_ind,
    "condnum", cond_num)

