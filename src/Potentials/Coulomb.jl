"""
    Coulomb Potential
"""
struct Coulomb <: Potential
    q_1::Float64
    q_2::Float64
    ε0::Float64
end

function Coulomb(params::Dict)
    #TODO
    return Coulomb(1.0, 1.0, 1.0)
end

function potential_energy(p::Coulomb, r::Position, args...)
    return p.q_1 * p.q_2 / (4.0 * π * p.ε0 * norm(r))
end


