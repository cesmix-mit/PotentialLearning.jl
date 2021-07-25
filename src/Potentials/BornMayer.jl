"""
    Born-Mayer Potential
"""
struct BornMayer <: Potential
    A::Float64
    ρ::Float64
end

function BornMayer(params::Dict)
    #TODO
    return BornMayer(1.0, 1.0)
end

function potential_energy(p::BornMayer, r::Position, args...)
    return p.A * exp(-norm(r) / p.ρ)
end

