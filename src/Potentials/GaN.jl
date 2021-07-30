"""
    GaN Potential

See 10.1088/1361-648X/ab6cbe
"""
mutable struct GaN <: Potential
    lj_Ga_Ga::LennardJones
    lj_N_N::LennardJones
    bm_Ga_N::BornMayer
    c::Coulomb
    no_Ga::Int64
    no_N::Int64
end

function GaN(params::Dict)
    # Creates the GaN model
    lj_Ga_Ga = LennardJones(params["ε_Ga_Ga"], params["σ_Ga_Ga"])
    lj_N_N = LennardJones(params["ε_N_N"], params["σ_N_N"])
    bm_Ga_N = BornMayer(params["A_Ga_N"], params["ρ_Ga_N"])
    c = Coulomb(params["q_Ga"], params["q_N"], params["ε0"])
    gan = GaN(lj_Ga_Ga, lj_N_N, bm_Ga_N, c, params["no_Ga"], params["no_N"])
    return gan
end

"""
    potential_energy(p::GaN, r::Position, args...)

Calculation of the potential energy between two atoms using the GaN model.
"""
function potential_energy(p::GaN, r::Position, args...)
    i = args[1]; j = args[2]
    if i <= p.no_Ga && j <= p.no_N # Ga-Ga interaction
        return potential_energy(p.c, r) + potential_energy(p.lj_Ga_Ga, r)
    elseif i > p.no_Ga && j > p.no_N # N-N interaction
        return potential_energy(p.c, r) + potential_energy(p.lj_N_N, r)
    else # Ga-N or N-Ga interaction
        return potential_energy(p.c, r) + potential_energy(p.bm_Ga_N, r)
    end

end



