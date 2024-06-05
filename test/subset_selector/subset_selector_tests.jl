using AtomsBase
using Unitful, UnitfulAtomic
using LinearAlgebra

# Initialize some fake descriptors
d = 8
num_atoms = 20
num_configs = 10
batch_size = 2
ld = [[randn(d) for i = 1:num_atoms] for j = 1:num_configs]
ld = LocalDescriptors.(ld)
ds = DataSet(Configuration.(ld))
gm = GlobalMean()
dp = DotProduct()

# kDPP tests
dpp = kDPP(ds, gm, dp; batch_size = batch_size)
@test typeof(dpp) <: SubsetSelector
@test typeof(get_random_subset(dpp)) <: Vector{<:Int}
@test typeof(get_dpp_mode(dpp)) <: Vector{<:Int}
@test typeof(get_inclusion_prob(dpp)) <: Vector{Float64}

# RandomSelector tests
r = RandomSelector(num_configs; batch_size = batch_size)
@test typeof(r) <: SubsetSelector
@test typeof(get_random_subset(r)) <: Vector{<:Int}

# DBSCANSelector tests
energy_units = u"eV"
distance_units = u"Å"
ds = load_data("../examples/data/Si-3Body-LAMMPS/data.xyz", ExtXYZ(energy_units, distance_units));
epsi, minpts, sample_size = 0.05, 5, batch_size
dbscans = DBSCANSelector(   ds,
                            epsi,
                            minpts,
                            sample_size)
@test typeof(dbscans) <: SubsetSelector
@test typeof(get_random_subset(dbscans)) <: Vector{<:Int}
