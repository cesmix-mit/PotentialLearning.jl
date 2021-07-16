using Base: Float64
using StaticArrays
const Position = SVector{3, Float64}
abstract type Potential end

"""
    Lennard-Jones Potential
"""
struct LennardJones <: Potential
    ε::Float64
    σ::Float64
end

function LennardJones(params::Dict)
    #ToDO
    return LennardJones(1.0, 1.0)
end

function potential_energy(r::Position, p::LennardJones)
    return 4.0 * p.ε * ((p.σ / norm(r))^12 - (p.σ / norm(r))^6)
end


"""
    Born-Mayer Potential
"""

struct BornMayer <: Potential
    A::Float64
    ρ::Float64
end

function BornMayer(params::Dict)
    #ToDO
    return BornMayer(1.0, 1.0)
end

function potential_energy(r::Position, p::BornMayer)
    return p.A * exp(-norm(r) / p.ρ)
end


"""
    Coulomb Potential
"""

struct Coulomb <: Potential
    q_1::Float64
    q_2::Float64
    ε0::Float64
end

function Coulomb(params::Dict)
    #ToDO
    return Coulomb(1.0, 1.0, 1.0)
end

function potential_energy(r::Position, p::Coulomb)
    return p.q_1 * p.q_2 / (4.0 * π * p.ε0 * norm(r))
end


"""
    GaN Potential
    See https://iopscience.iop.org/article/10.1088/1361-648X/ab6cbe
"""
struct GaN <: Potential
    lj_Ga_Ga::LennardJones
    lj_N_N::LennardJones
    bm_Ga_N::BornMayer
    c::Coulomb
    no_Ga::Int64
    no_N::Int64
end

"""
    GaN(params::Dict)
    
Creates 
"""
function GaN(params::Dict)
    # Read parameters from a configuration file
    GaN_params = Dict()
    path = params["path"]
    open(string(path, "/GaN.params")) do f
        while !eof(f)
            line = split(readline(f))
            GaN_params[line[1]] = parse(Float64, line[2])
        end
    end 
    # Creates the GaN model
    lj_Ga_Ga = LennardJones(GaN_params["ε_Ga_Ga"], GaN_params["σ_Ga_Ga"])
    lj_N_N = LennardJones(GaN_params["ε_N_N"], GaN_params["σ_N_N"])
    bm_Ga_N = BornMayer(GaN_params["A_Ga_N"], GaN_params["ρ_Ga_N"])
    c = Coulomb(GaN_params["q_Ga"], GaN_params["q_N"], GaN_params["ε0"])
    gan = GaN(lj_Ga_Ga, lj_N_N, bm_Ga_N, c, GaN_params["no_Ga"], GaN_params["no_N"])
    return gan
end

function potential_energy(i::Int64, j::Int64, r::Position, p::GaN)
    if i <= p.no_Ga && j <= p.no_N # Ga-Ga interaction
        return potential_energy(r, p.c) + potential_energy(r, p.lj_Ga_Ga)
    elseif i > p.no_Ga && j > p.no_N # N-N interaction
        return potential_energy(r, p.c) + potential_energy(r, p.lj_N_N)
    else # Ga-N or N-Ga interaction
        return potential_energy(r, p.c) + potential_energy(r, p.bm_Ga_N)
    end
end



