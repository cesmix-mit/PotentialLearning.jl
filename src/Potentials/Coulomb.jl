"""
    Coulomb Potential
"""
struct Coulomb <: Potential
    ε0::Float64
    q_1::Float64
    q_2::Float64
end

function Coulomb(params::Dict)
    #TODO: read configuration file
    ε0 = params["ε0"]
    q_1 = params["q_1"]
    q_2 = params["q_2"]
    return Coulomb(ε0, q_1, q_2)
end

function potential_energy(p::Coulomb, r::Position, args...)
    return p.q_1 * p.q_2 / (4.0 * π * p.ε0 * norm(r))
end


