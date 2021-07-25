"""
    Ziegler-Biersack-Littmark (ZBL) Potential
    
See https://docs.lammps.org/pair_zbl.html
"""
struct ZBL <: Potential
    ε0::Float64
    zi::Float64
    zj::Float64
end

function ZBL(params::Dict)
    #TODO
    return ZBL(1.0, 1.0, 1.0)
end

function potential_energy(p::ZBL, r::Position, args...)
    #TODO
    return 1.0
end

