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

## UnivariateLinearProblem ###############################################################

# initialize some fake energy descriptors
d = 8
num_atoms = 20
num_configs = 10
ld_0 = [[randn(d) for i = 1:num_atoms] for j = 1:num_configs]
ld = LocalDescriptors.(ld_0)
e = Energy.(rand(num_configs), (u"eV",))
ds = DataSet(Configuration.(e, ld))


# test input types
lp = PotentialLearning.LinearProblem(ds)
@test typeof(lp) <: PotentialLearning.LinearProblem
@test typeof(lp) <: PotentialLearning.UnivariateLinearProblem
@test typeof(ds) <: DataSet
@test typeof(lb) <: InteratomicPotentials.LinearBasisPotential


# test input values before learning
@test lp.dv_data == get_values.(e)
@test lp.iv_data == sum.(get_values.(get_local_descriptors.(ds)))
@test lp.σ == [1.0]
@test lp.β == zeros(d) # coeffs in LinearProblem
@test lb.β == zeros(d) # coeffs in LinearBasisPotential


# test learning functions
# test learn!(lb::LinearBasisPotential, ds::DataSet; α::Real = 1e-8, return_cov = true)
α = 1e-8
lb, Σ = learn!(lb, ds, α)
# test learn!(lp::UnivariateLinearProblem; α = 1e-8) (internal method)
learn!(lp, α) 
# test the two give the same output
@test lb.β == lp.β



## CovariateLinearProblem ###############################################################

# initialize some fake energy and force descriptors
# ld_0 = [[randn(d) for i = 1:num_atoms] for j = 1:num_configs]
# fd_0 = [[[randn(d), randn(d), randn(d)] for i = 1:num_atoms] for j = 1:num_configs]
# ld = LocalDescriptors.(ld_0)
# fd = ForceDescriptors.(fd_0)
# e = Energy.(rand(num_configs), (u"eV",))
# f = Forces.()
# u"eV/Å"
# ds = DataSet(Configuration.(e, ld))
