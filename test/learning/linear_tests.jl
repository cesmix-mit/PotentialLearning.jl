using Unitful, UnitfulAtomic
using InteratomicPotentials, InteratomicBasisPotentials

ace = ACE(species = [:Na],         # species
          body_order = 2,          # 4-body
          polynomial_degree = 8,   # 8 degree polynomials
          wL = 1.0,                # Defaults, See ACE.jl documentation 
          csp = 1.0,               # Defaults, See ACE.jl documentation 
          r0 = 1.0,                # minimum distance between atoms
          rcutoff = 5.0)           # cutoff radius 
lb = LBasisPotentialExt(ace)

# UnivariateLinearProblem ######################################################

# Initialize some fake energy descriptors
d = 8
num_atoms = 20
num_configs = 8
ld_0 = [[randn(d) for i = 1:num_atoms] for j = 1:num_configs]
ld = LocalDescriptors.(ld_0)
e = Energy.(rand(num_configs), (u"eV",))
ds = DataSet(Configuration.(e, ld))

# Test input types
lp = PotentialLearning.LinearProblem(ds)
@test typeof(lp) <: PotentialLearning.LinearProblem
@test typeof(lp) <: PotentialLearning.UnivariateLinearProblem
@test typeof(ds) <: DataSet
@test typeof(lb) <: InteratomicPotentials.LinearBasisPotential

# Test input values before learning
@test lp.dv_data == get_values.(e)
@test lp.iv_data == sum.(get_values.(get_local_descriptors.(ds)))
@test lp.σ == [1.0]
@test lp.β == zeros(d)
@test lp.β0 == zeros(1)
@test lb.β == zeros(d)
@test lb.β0 == zeros(1)

# Test learning functions based on weighted least squares approach
learn!(lb, ds) # default
ws, int = [1.0], false
learn!(lp, ws, int)
@test lb.β == lp.β
@test lb.β0 == lp.β0
@test maximum(abs.(get_all_energies(ds, lb) - lp.dv_data)) < 1.e-4

ws, int = [1.0], true
learn!(lb, ds, ws, int)
learn!(lp, ws, int)
@test lb.β == lp.β
@test lb.β0 == lp.β0
@test maximum(abs.(get_all_energies(ds, lb) - lp.dv_data)) < 1.e-4

# Test learning functions based on maximum likelihood estimation approach
α = 1e-8
Σ = learn!(lb, ds, α)
learn!(lp, α) 
@test lb.β == lp.β
@test maximum(abs.(get_all_energies(ds, lb) - lp.dv_data)) < 1.e-4

# CovariateLinearProblem #######################################################

# TODO:
# initialize some fake energy and force descriptors
# ld_0 = [[randn(d) for i = 1:num_atoms] for j = 1:num_configs]
# fd_0 = [[[randn(d), randn(d), randn(d)] for i = 1:num_atoms] for j = 1:num_configs]
# ld = LocalDescriptors.(ld_0)
# fd = ForceDescriptors.(fd_0)
# e = Energy.(rand(num_configs), (u"eV",))
# f = Forces.()
# u"eV/Å"
# ds = DataSet(Configuration.(e, ld))
