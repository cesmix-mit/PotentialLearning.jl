using LinearAlgebra
using AtomsBase, Unitful, UnitfulAtomic, StaticArrays
using InteratomicPotentials, InteratomicBasisPotentials
using ACE1, JuLIP


# initialize a LinearBasisPotential
n_body = 2  
max_deg = 8 
r0 = 1.0 
rcutoff = 5.0 
wL = 1.0 
csp = 1.0 
ace = ACE([:Na], n_body, max_deg, wL, csp, r0, rcutoff)
lb = LBasisPotential(ace)


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
lb, Σ = learn!(lb, ds)
# test learn!(lp::UnivariateLinearProblem; α = 1e-8) (internal method)
lp = learn!(lp) 
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