################################################################################
# Lennard-Jones Potential
################################################################################

struct LennardJones
    ε::Float64
    σ::Float64
end

potential_energy(r::Point, p::LennardJones) =
    4.0 * p.ε * ((p.σ / norm(r))^12 - (p.σ / norm(r))^6)


################################################################################
# Born-Mayer Potential
################################################################################

struct BornMayer
    A::Float64
    ρ::Float64
end

potential_energy(r::Point, p::BornMayer) = p.A * exp(-norm(r) / p.ρ)


################################################################################
# Coulomb Potential
################################################################################

struct Coulomb
    q_1::Float64
    q_2::Float64
    ε0::Float64
end

potential_energy(r::Point, p::Coulomb) =
    p.q_1 * p.q_2 / (4.0 * π * p.ε0 * norm(r))


################################################################################
# GaN Potential. 
# See https://iopscience.iop.org/article/10.1088/1361-648X/ab6cbe
################################################################################

struct GaN
    lj_Ga_Ga::LennardJones
    lj_N_N::LennardJones
    bm_Ga_N::BornMayer
    c::Coulomb
    no_Ga::Int64
    no_N::Int64
end

potential_energy(i, j, r::Point, p::GaN) =
    if i <= p.no_Ga && j <= p.no_N # Ga-Ga interaction
        return potential_energy(r, p.c) + potential_energy(r, p.lj_Ga_Ga)
    elseif i > p.no_Ga && j > p.no_N # N-N interaction
        return potential_energy(r, p.c) + potential_energy(r, p.lj_N_N)
    else # Ga-N or N-Ga interaction
        return potential_energy(r, p.c) + potential_energy(r, p.bm_Ga_N)
    end
    
function load_GaN(path)
    params = Dict()
    open(string(path, "/GaN.params")) do f
        while !eof(f)
            line = split(readline(f))
            params[line[1]] = parse(Float64, line[2])
        end
    end 
    lj_Ga_Ga = LennardJones(params["ε_Ga_Ga"], params["σ_Ga_Ga"])
    lj_N_N = LennardJones(params["ε_N_N"], params["σ_N_N"])
    bm_Ga_N = BornMayer(params["A_Ga_N"], params["ρ_Ga_N"])
    c = Coulomb(params["q_Ga"], params["q_N"], params["ε0"])
    gan = GaN(lj_Ga_Ga, lj_N_N, bm_Ga_N, c, params["no_Ga"], params["no_N"])
    return gan
end





