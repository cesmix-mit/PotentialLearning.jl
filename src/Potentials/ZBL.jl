"""
    Ziegler-Biersack-Littmark (ZBL) Potential
    
See https://docs.lammps.org/pair_zbl.html
"""
struct ZBL <: Potential
    ε0::Float64
    e::Float64
    zi::Float64
    zj::Float64
    a::Float64
end

function ZBL(params::Dict)
    ε0 = params["ε0"]
    e = params["e"]
    zi = params["zi"]
    zj = params["zj"]
    a = 0.46850 / (zi^0.23 + zj^0.23)
    return ZBL(1.0, 1.0, 1.0, 1.0, a)
end

function ϕ(p::ZBL, r::Float64)
    return   0.18175 * p.e^(-3.1998*x) + 0.50986*p.e^(-0.94229*x) +
             0.28022*p.e^(-0.40290*x) + 0.2817*p.e^(-0.20162*x)
end

function S(r)
#TODO
#    A = 
#    B = 
#    C = 
#    if r < r1
#        return C
#    else
#        return 
#    end
    return 1.0
end

function potential_energy(p::ZBL, r::Position, args...)
    return 1 / (4.0 * π * p.ε0) * p.zi * p.zj * p.e^2 / norm(r) *
           ϕ(p, norm(r)/a) + S(norm(r))
end

