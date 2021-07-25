using Base: Float64
using Zygote
using StaticArrays
const Position = SVector{3, Float64}
const Force = SVector{3, Float64}
abstract type Potential end

include("LennardJones.jl")
include("BornMayer.jl")
include("Coulomb.jl")
include("GaN.jl")

"""
    atom_type(i::Int64)
    
Returns the atom type of the i-th atom of the configuration.
"""
function atom_type(i::Int64)
    #TODO
    return i
end

"""
    potential_energy(p::Potential, atomic_positions::Vector{Position}, rcut::Float64)

Calculation of the potential energy of a particular atomic configuration.
It is based on the atomic positions of the configuration, the rcut, and a
particular potential.
"""
function potential_energy(p::Potential, atomic_positions::Vector{Position}, rcut::Float64)
    acc = 0.0
    for i = 1:length(atomic_positions)
        for j = i:length(atomic_positions)
            r_diff = (atomic_positions[i] - atomic_positions[j])
            if norm(r_diff) <= rcut && norm(r_diff) > 0.0
                acc += potential_energy(p, r_diff, atom_type(i), atom_type(j))
            end
        end
    end
    return acc
end


"""
    forces(p::Potential, atomic_positions::Vector{Position}, rcut::Float64)

Calculation of the energy of the forces of each atom in each atomic configuration.
"""
function forces(p::Potential, atomic_positions::Vector{Position}, rcut::Float64)
    forces = Vector{Force}()
    for i = 1:length(atomic_positions)
        f_i = Force(0.0, 0.0, 0.0)
        for j = 1:length(atomic_positions)
            r_diff = atomic_positions[i] - atomic_positions[j]
            if norm(r_diff) <= rcut && norm(r_diff) > 0.0
                ∇potential_energy(p, r, i, j) =
                     gradient(r -> potential_energy(p, r_diff + r, i, j), r)[1]
                f_ij = -∇potential_energy(p, atomic_positions[i], atom_type(i),
                                                                  atom_type(j))
                f_i += f_ij
            end
        end
        push!(forces, f_i)
    end
    return forces
end


