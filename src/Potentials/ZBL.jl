"""
    Ziegler-Biersack-Littmark (ZBL) Potential
    
See https://docs.lammps.org/pair_zbl.html
"""
mutable struct ZBL <: Potential
    ε0::Float64      # Electrical permittivity of vacuum.
    e::Float64       # Electron charge.
    zi::Float64      # Atomic number of the atom i.
    zj::Float64      # Atomic number of the atom j.
    a::Float64
    rcutfac::Float64 # Rcut
end

"""
    ZBL(params::Dict)
    
Creates a ZBL potential.
"""
function ZBL(params::Dict)
    # Creates the ZBL model
    ε0 = params["ε0"]
    e = params["e"]
    zi = params["zi"]
    zj = params["zj"]
    a = 0.46850 / (zi^0.23 + zj^0.23)
    rcutfac = params["rcutfac"]
    return ZBL(ε0, e, zi, zj, a, rcutfac)
end

"""
    potential_energy(p::ZBL, r::Position, args...)
    
Calculates the potential energy of ZBL with a switching function.
"""
function potential_energy(p::ZBL, r::Position, args...)
    return  zbl(p, norm(r)) + S(p, norm(r))
end

"""
    zbl(p::ZBL, r::Float64)
    
ZBL function without the switching function S.
"""
function zbl(p::ZBL, r::Float64)
    return 1 / (4.0 * π * p.ε0) * p.zi * p.zj * p.e^2 / r * ϕ(p, r / p.a)
end

"""
    S(r)

Switching function that ramps the energy, force, and curvature smoothly to zero
between an inner and outer cutoff. Here, the inner and outer cutoff are the same
for all pairs of atom types.
"""
function S(p::ZBL, r::Float64)
    return r <= p.rcutfac ? 0.5 * (cos(pi * r / p.rcutfac) + 1.0) : 0
end

# TODO: check this function: "exp" or "e"?
"""
    ϕ(p::ZBL, x::Float64)
    
Auxiliary function, necessary to calculate ZBL.
"""
function ϕ(p::ZBL, x::Float64)
    return   0.18175 * exp(-3.1998*x)  + 0.50986 * exp(-0.94229*x) +
             0.28022 * exp(-0.40290*x) + 0.2817  * exp(-0.20162*x)
end


