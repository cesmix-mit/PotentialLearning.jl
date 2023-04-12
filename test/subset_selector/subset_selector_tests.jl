using AtomsBase
using Unitful, UnitfulAtomic
using LinearAlgebra
# initialize some fake descriptors
d = 8
num_atoms = 20
num_configs = 10
batch_size = 2
ld = [[randn(d) for i = 1:num_atoms] for j = 1:num_configs]
ld = LocalDescriptors.(ld)
ds = DataSet(Configuration.(ld))

gm = GlobalMean()
dp = DotProduct()

dpp = kDPP(ds, gm, dp; batch_size = batch_size)
@test typeof(dpp) <: SubsetSelector
@test typeof(get_random_subset(dpp)) <: Vector{<:Int}
@test typeof(get_dpp_mode(dpp)) <: Vector{<:Int}
@test typeof(get_inclusion_prob(dpp)) <: Vector{Float64}

r = RandomSelector(num_configs; batch_size = batch_size)
@test typeof(r) <: SubsetSelector
@test typeof(get_random_subset(r)) <: Vector{<:Int}
