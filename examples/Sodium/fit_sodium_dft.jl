using LinearAlgebra, Random, Statistics, StatsBase, Distributions
using AtomsBase, Unitful, UnitfulAtomic, StaticArrays
using InteratomicPotentials, InteratomicBasisPotentials
using CairoMakie
using JLD
using DPP
include("./src/PotentialLearning.jl")
using .PotentialLearning

#################### Importing Data ###################
# Import Raw Data

energies, forces = JLD.load("examples/Sodium/data/liquify_sodium_dftk_calculations.jld", "energies", "forces");
e = [ ustrip(uconvert(u"eV", sum(collect(values(di))) * u"hartree")) for di in energies];
f = [ [ ustrip.(uconvert.(u"eV/Å", [fij[1], fij[2], fij[3]] * u"hartree/bohr")) for fij in fi] for fi in forces ];

e = Energy.(e);
f = Forces.(f, (u"eV/Å"));

# Import configurations 
ds_temp, _ = load_data("examples/Sodium/data/liquify_sodium.yaml", YAML(u"eV", u"Å"));
systems = get_system.(ds_temp);

# Only take subset 
ind = 200
ds_full = DataSet(Configuration.(systems[ind:end], e[ind:end], f[ind:end]));

rand_inds = randperm(length(ds_full))[1:400];
ds_train = ds_full[rand_inds];

# Compute descriptors
n_body = 4  # 2-body
max_deg = 8 # 8 degree polynomials
r0 = 1.0 # minimum distance between atoms
rcutoff = 5.0 # cutoff radius 
wL = 1.0 # Defaults, See ACE.jl documentation 
csp = 1.0 # Defaults, See ACE.jl documentation
ace = ACE([:Na], n_body, max_deg, wL, csp, r0, rcutoff)

descriptors, force_descriptors = JLD.load("examples/Sodium/data/400_train_descriptors_force_descriptors.jld", "descriptors", "force_descriptors");
descriptors = LocalDescriptors.(descriptors);
force_descriptors = ForceDescriptors.(force_descriptors);

ds_train_e = DataSet( Configuration.(get_energy.(ds_train), descriptors) )
ds_train_f = DataSet( Configuration.(get_forces.(ds_train), force_descriptors))
ds_train_e_f = DataSet( Configuration.(get_energy.(ds_train), descriptors, get_forces.(ds_train), force_descriptors))

lp_e = LinearProblem(ds_train_e);
println(typeof(lp_e))
learn!(lp_e);

lp_f = LinearProblem(ds_train_f);
println(typeof(lp_f))
learn!(lp_f);

lp_e_f = LinearProblem(ds_train_e_f);
println(typeof(lp_e_f))
learn!(lp_e_f);










############################################################################################################
############################################################################################################

descriptors = LocalDescriptors[]
force_descriptors = ForceDescriptors[]
for (i, sys) in enumerate(get_system.(ds_train))
    if i % 20 == 0
        println("System #$i out of 400")
    end
    push!(descriptors, LocalDescriptors(InteratomicBasisPotentials.get_local_descriptors(sys, ace)))
    ftemp = InteratomicBasisPotentials.get_force_descriptors(sys, ace)
    ftemp = [ [fi[i, :] for i = 1:3] for fi in ftemp ]
    push!(force_descriptors, ForceDescriptors(ftemp))
end
JLD.save("examples/Sodium/data/400_train_descriptors_force_descriptors.jld", "descriptors", get_values.(descriptors), "force_descriptors", get_values.(force_descriptors) )