"""
    Lennard-Jones Potential
"""
struct LennardJones <: Potential
    ε::Float64
    σ::Float64
end

function LennardJones(params::Dict)
    #TODO
    return LennardJones(1.0, 1.0)
end

function potential_energy(p::LennardJones, r::Position, args...)
    return 4.0 * p.ε * ((p.σ / norm(r))^12 - (p.σ / norm(r))^6)
end

