#] activate ../../ElectronicStructure.jl/
#] activate ../

using Unitful, PeriodicTable, StaticArrays, LinearAlgebra
using AtomsBase
using InteratomicPotentials
using ElectronicStructure
using .PotentialLearning

################################################################################
# InteratomicPotentials.jl #####################################################
################################################################################

# Modifications to InteratomicPotentials.jl
# TODO: Discuss parametric types {D}

abstract type ArbitraryPotential{D} end
abstract type EmpiricalPotential{D} <: ArbitraryPotential{D} end

# TODO: Unitful.Energy, Unitful.Length
mutable struct LennardJones{D} <: EmpiricalPotential{D}
    ϵ::Unitful.Energy
    σ::Unitful.Length
end

function potential_energy(r::SVector, p::LennardJones)
    d = p.σ / norm(r)
    return 4.0 * p.ϵ * ( d^12 - d^6 )
end

# TODO: New functions needed in "gen_test_data"

function potential_energy(s::AbstractSystem, p::ArbitraryPotential)
    N = size(s)[1]
    return sum([potential_energy(position(getindex(s,i)) - position(getindex(s,j)), p)
                for i in 1:N for j in i+1:N])
end

function forces(s::AbstractSystem{D}, p::ArbitraryPotential{D}) where {D}
    N = size(s)[1]
    return [ SVector{D}(zeros(D) * 1.0u"N") for i in 1:N ]
end

function virial(s::AbstractSystem{D}, p::ArbitraryPotential{D}) where {D}
    return 0.0
end

function virial_stress(s::AbstractSystem{D}, p::ArbitraryPotential{D}) where {D}
    return SMatrix{D, D}(zeros(D, D))
end

#function SNAP(rcutfac::Float64, twojmax::Int, num_atom_types::Int)
#    keywords = SNAPkeywords(0, 0, 0, 0, 0)
#    num_coeffs = get_num_coeffs(twojmax)
#    return SNAP(zeros(num_atom_types * num_coeffs + 1), rcutfac, twojmax, keywords)
#end

function inter_pot_conf(atomic_confs::Vector)
    atom_names = nothing
    radii = [3.5]
    weights = [1.0]
    boundary_type = ["p", "p", "p"]
    units = "lj"
    v = zeros(D)
    new_atomic_confs = []
    for c in atomic_confs

        atoms = Vector{Atom}(undef, 0)
        angles = Vector{Angle}(undef, 0)
        bonds = Vector{Bond}(undef, 0)
        impropers = Vector{Improper}(undef, 0)
        dihedrals = Vector{Dihedral}(undef, 0)

        num_atoms           = length(c)
        num_atom_types      = length(unique(species.(c)))
        num_bond_types      = 0
        num_angle_types     = 0
        num_dihedral_types  = 0
        num_improper_types  = 0

        x_bounds = [0.0, bounding_box(c)[1][1].val]
        y_bounds = [0.0, bounding_box(c)[2][2].val]
        z_bounds = [0.0, bounding_box(c)[3][3].val]
        
        domain = Domain(SVector{D}([x_bounds, y_bounds, z_bounds]), SVector{D}(boundary_type))

        if length(radii) != num_atom_types
            radii = radii[1] .* ones(num_atom_types)
        end
        if length(weights) != num_atom_types
            weights = weights .* ones(num_atom_types)
        end

        masses = unique(atomic_mass.(c))

        if atom_names == nothing
            atom_names = [Symbol(i) for i = 1:num_atom_types]
        end

        atoms = []
        for (s, p) in zip(species(c), position(c))
            p2 = [p[1].val, p[2].val, p[3].val]
            push!(atoms, Atom(s.atomic_mass.val, p2, v))
        end

        new_c = Configuration(atoms, num_atom_types,
                              angles, num_angle_types,
                              bonds, num_bond_types,
                              impropers, num_improper_types,
                              dihedrals, num_dihedral_types,
                              atom_names, masses,
                              radii, weights,
                              domain, units)
        push!(new_atomic_confs, new_c)
    end
    return new_atomic_confs
end



################################################################################
# Learning example #############################################################
################################################################################

"""
    gen_test_atomic_conf(D)

Generate test atomic configurations
"""
function gen_test_atomic_conf(D, L)
    # Domain
    box = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * L
    # Boundary conditions
    bcs = [Periodic(), Periodic(), DirichletZero()]
    # No. of atoms per configuration
    N = 100
    # No. of configurations
    M = 20 
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
D = 3; L = 1.0u"cm"; E = 1.0u"J"

# Generate atomic configurations: domain and particles (position, velocity, etc)
atomic_confs = gen_test_atomic_conf(D, L)

# Generate learning data
lj = LennardJones{D}(1.657e-21u"J", 0.34u"nm")
data = gen_test_data(D, atomic_confs, lj)

# Define potential
rcutfac = 0.1; twojmax = 2
inter_pot_atomic_confs = inter_pot_conf(atomic_confs)
snap = SNAP(rcutfac, twojmax, inter_pot_atomic_confs[1])

# Define learning problem
lp = SmallSNAPLP{D}(snap, data)  #HINT: get_snap(r::Vector{Configuration}, p; dim = 3, reference = true)

# Learn :-)
#learn(lp, LeastSquaresOpt{T}())



