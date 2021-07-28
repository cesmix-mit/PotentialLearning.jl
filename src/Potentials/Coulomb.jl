"""
    Coulomb Potential
"""
mutable struct Coulomb <: Potential
    q_1::Float64
    q_2::Float64
    ε0::Float64
end

function Coulomb(params::Dict)
    #TODO: read configuration file
    q_1 = params["q_1"]
    q_2 = params["q_2"]
    ε0 = params["ε0"]
    return Coulomb(q_1, q_2, ε0)
end

function potential_energy(p::Coulomb, r::Position, args...)
    return p.q_1 * p.q_2 / (4.0 * π * p.ε0 * norm(r))
end


