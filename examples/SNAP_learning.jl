using AtomsBase, DFT, InteratomicPotentials, PotentialLearning

function gen_atomic_geom()
  box = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]u"m"
  bcs = [Periodic(), Periodic(), DirichletZero()]
  positions = [0.25 0.25 0.25; 0.75 0.75 0.75]u"m"
  elems = [elements[:C], elements[:C]]

  atom1 = StaticAtom(SVector{3}(positions[1,:]),elems[1])
  atom2 = StaticAtom(SVector{3}(positions[2,:]),elems[2])

  aos = FlexibleSystem(box, bcs, [atom1, atom2])
  #soa = FastSystem(box, bcs, positions, elems)
  return [aos, aos]
end

function gen_test_data(p::InteratomicPotential{D, T}
                       atomic_confs::Vector{AbstractSystem{D, T}})
  energies = [ energy(a, p) for s in atomic_confs ]
  forces   = [ forces(a, p) for s in atomic_confs ]
  stresses = [ stresses(a, p) for s in atomic_confs ]
  data = SmallDFTData{D, T}(energies, forces, stresses)
  return data
end

# Define parametric types
D = 3; T = Float32

# Generate atomic configurations: domain and particles (position, velocity, etc)
atomic_confs = gen_atomic_conf()

# Generate learning data
lj = LennardJones{D, T}(1.657e-21u"J", 0.34u"nm")
data = gen_test_data(lj, atomic_confs) 

# Define potential
snap = SNAP{D, T}(atomic_confs)

# Define learning problem
lp = SmallSNAPLP{D, T}(snap, data)

# Learn :-)
learn(lp, LeastSquaresOpt{T}())
