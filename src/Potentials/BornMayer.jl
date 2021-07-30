"""
    Born-Mayer Potential
"""
mutable struct BornMayer <: Potential
    A::Float64
    ρ::Float64
end

"""
    BornMayer(params::Dict)
    
Creates a BM potential.
"""
function BornMayer(params::Dict)
    A = params["A"]
    ρ = params["ρ"]
    return BornMayer(A, ρ)
end

"""
    potential_energy(p::BornMayer, r::Position, args...)
    
Calculates Born-Mayer potential energy.
"""
function potential_energy(p::BornMayer, r::Position, args...)
    return p.A * exp(-norm(r) / p.ρ)
end

