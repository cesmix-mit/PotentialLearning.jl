# SNAP learning/fitting example
#
# It shows a first integration of the following packages under development:
# AtomsBase.jl, ElectronicStructure.jl, InteratomicPotentials.jl, and PotentialLearning.jl
#

import Pkg
Pkg.add("Unitful")
Pkg.add("PeriodicTable")
Pkg.add("StaticArrays")
Pkg.add("LinearAlgebra")
Pkg.add("AtomsBase")
Pkg.add(url="git@github.com:cesmix-mit/ElectronicStructure.jl.git")
Pkg.add(url="https://github.com/cesmix-mit/InteratomicPotentials.jl.git", rev="integrated-branch")
Pkg.add(url="https://github.com/cesmix-mit/PotentialLearning.jl.git", rev="refactor")

using Unitful, PeriodicTable, StaticArrays, LinearAlgebra
using AtomsBase
using ElectronicStructure
using InteratomicPotentials
using PotentialLearning

"""
    gen_test_atomic_conf(D)

Generate test atomic configurations
"""
function gen_test_atomic_conf(D)
    # Domain
    box = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]] * 1.0u"cm"
    # Boundary conditions
    bcs = [Periodic(), Periodic(), DirichletZero()]
    # No. of atoms per configuration
    N = 30
    # No. of configurations
    M = 20
    # Element
    c = elements[:C]
    # Define atomic configurations
    atomic_confs = []
    for j in 1:M
        atoms = []
        for i in 1:N
            pos = SVector{D}(rand(D)*L...)
            atom = StaticAtom(pos,c)
            push!(atoms, atom)
        end
        push!(atomic_confs, FlexibleSystem(box, bcs, atoms))
    end
    return atomic_confs
end

# Define parametric types 
D = 3, T = Float64 # TODO: discuss which parametric types are necessary, define a common policy for all packages 

# Generate test atomic configurations: domain and particles (position, velocity, etc)
atomic_confs = gen_test_atomic_conf(D)

# Generate learning data using Lennard Jones and the atomic configurations
lj = LennardJones(1.657e-21u"J", 0.34u"nm")
data = gen_test_data(D, atomic_confs, lj)

# Define target potential: SNAP
rcutfac = 0.1; twojmax = 2
inter_pot_atomic_confs = inter_pot_conf(atomic_confs) # TODO: remove after full integration with InteratomicPotentials.jl
snap = SNAP(rcutfac, twojmax, inter_pot_atomic_confs[1]) #TODO: improve interface, do not send a conf as argument

# Define learning problem
lp = SmallSNAPLP(snap, inter_pot_atomic_confs, data)

# Learn :-)
learn(lp, LeastSquaresOpt{D, T}())



