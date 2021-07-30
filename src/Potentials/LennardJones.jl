"""
    Lennard-Jones Potential
"""
mutable struct LennardJones <: Potential
    ε::Float64
    σ::Float64
end

"""
    LennardJones(params::Dict)
    
Creates the LennardJones potential.
"""
function LennardJones(params::Dict)
    # Creates the LennardJones model
    ε = params["ε"]
    σ = params["σ"]
    return LennardJones(ε, σ)
end

"""
    potential_energy(p::LennardJones, r::Position, args...)
    
Calculates LennardJones potential energy.
"""
function potential_energy(p::LennardJones, r::Position, args...)
    return 4.0 * p.ε * ((p.σ / norm(r))^12 - (p.σ / norm(r))^6)
end

