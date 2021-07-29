"""
    Born-Mayer Potential
"""
mutable struct BornMayer <: Potential
    A::Float64
    ρ::Float64
end

function BornMayer(params::Dict)
    # Read parameters from a configuration file
    BM_params = load_params(string(params["path"], "/BornMayer.conf"))
    # Creates the BM model
    A = BM_params["A"]
    ρ = BM_params["ρ"]
    return BornMayer(A, ρ)
end

function potential_energy(p::BornMayer, r::Position, args...)
    return p.A * exp(-norm(r) / p.ρ)
end

