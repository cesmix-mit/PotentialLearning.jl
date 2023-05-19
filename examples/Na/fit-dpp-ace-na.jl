push!(Base.LOAD_PATH, "../../")

using AtomsBase
using Unitful, UnitfulAtomic
using InteratomicPotentials
using InteratomicBasisPotentials
using PotentialLearning
#using JLD

# Load dataset
confs, thermo = load_data("data/liquify_sodium.yaml", YAML(:Na, u"eV", u"Å"))
confs, thermo = confs[220:end], thermo[220:end]

# Split dataset
conf_train, conf_test = confs[1:1000], confs[1001:end]

# Define ACE
ace = ACE(species = [:Na],         # species
          body_order = 4,          # 4-body
          polynomial_degree = 8,   # 8 degree polynomials
          wL = 1.0,                # Defaults, See ACE.jl documentation 
          csp = 1.0,               # Defaults, See ACE.jl documentation 
          r0 = 1.0,                # minimum distance between atoms
          rcutoff = 5.0)           # cutoff radius 

# Update training dataset by adding energy (local) descriptors
e_descr_train = compute_local_descriptors(conf_train, ace)
#e_descr_train = JLD.load("data/sodium_empirical_full.jld", "descriptors")
ds_train = DataSet(conf_train .+ e_descr_train)

# Learn, using DPP
lb = LBasisPotential(ace)
dpp = kDPP(ds_train, GlobalMean(), DotProduct(); batch_size = 200)
dpp_inds = get_random_subset(dpp)
lb, Σ = learn!(lb, ds_train[dpp_inds]; α = 1e-6)

# Post-process output

# Update test dataset by adding energy and force descriptors
e_descr_test = compute_local_descriptors(conf_test, ace)
ds_test = DataSet(conf_test .+ e_descr_test)

# Get true and predicted values
e_train, e_train_pred = get_all_energies(ds_train), get_all_energies(ds_train, lb)
e_test, e_test_pred   = get_all_energies(ds_test), get_all_energies(ds_test, lb)

# Compute metrics
e_mae, e_rmse, e_rsq = calc_metrics(e_train, e_train_pred)


