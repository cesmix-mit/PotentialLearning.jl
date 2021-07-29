"""
    Coulomb Potential
"""
mutable struct Coulomb <: Potential
    q_1::Float64
    q_2::Float64
    ε0::Float64
end

function Coulomb(params::Dict)
    # Read parameters from a configuration file
    BM_params = load_params(string(params["path"], "/Coulomb.conf"))
    # Creates the Coulomb model
    q_1 = BM_params["q_1"]
    q_2 = BM_params["q_2"]
    ε0 = BM_params["ε0"]
    return Coulomb(q_1, q_2, ε0)
end

function potential_energy(p::Coulomb, r::Position, args...)
    return p.q_1 * p.q_2 / (4.0 * π * p.ε0 * norm(r))
end


