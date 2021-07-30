"""
    Born-Mayer Potential
"""
mutable struct BornMayer <: Potential
    A::Float64
    ρ::Float64
end

function BornMayer(params::Dict)
    # Creates the BM model
    A = params["A"]
    ρ = params["ρ"]
    return BornMayer(A, ρ)
end

function potential_energy(p::BornMayer, r::Position, args...)
    return p.A * exp(-norm(r) / p.ρ)
end

