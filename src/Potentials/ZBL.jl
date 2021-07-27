"""
    Ziegler-Biersack-Littmark (ZBL) Potential
    
See https://docs.lammps.org/pair_zbl.html
"""
struct ZBL <: Potential
    ε0::Float64      # Electrical permittivity of vacuum.
    e::Float64       # Electron charge.
    zi::Float64      # Nuclear charges of the atom i.
    zj::Float64      # Nuclear charges of the atom j.
    a::Float64
    r_inner::Float64 # Inner cutoff. Distance where switching function begins.
    r_outer::Float64 # Outer cutoff. Global cutoff for ZBL interaction.
    A::Float64
    B::Float64
    C::Float64
end

function ZBL(params::Dict)

    #TODO: read configuration file
    ε0 = params["ε0"]
    e = params["e"]
    zi = params["zi"]
    zj = params["zj"]
    a = 0.46850 / (zi^0.23 + zj^0.23)
    r_inner = params["r_inner"]
    r_outer = params["r_outer"]
    
    p = ZBL(ε0, e, zi, zj, a, r_inner, r_outer, 0.0, 0.0, 0.0)
    
    zbl′(p, r) = gradient(r -> zbl(p, r), r)[1]
    zbl′′(p, r) = gradient(r -> zbl′(p, r), r)[1]
    
    # See https://docs.lammps.org/pair_zbl.html
    #     https://docs.lammps.org/pair_gromacs.html
    #
    #   E(r_outer) = zbl(p, r_outer) + S(r_outer)
    #   S(r_outer) = -E(r_outer)
    #   S′(r_outer) = -E′(r_outer)
    #   S′′(r_outer) = -E′′(r_outer)
    #   =>
    E(r) = -zbl(p, r) / 2.0
    E′(r) = -zbl′(p, r) / 2.0
    E′′(r) = -zbl′′(p, r) / 2.0
    
    p.A = (-3.0 * E′(r_outer) + (r_outer - r_inner) * E′′(r_outer)) / (r_outer - r_inner)^2
    p.B = (-2.0 * E′(r_outer) - (r_outer - r_inner) * E′′(r_outer)) / (r_outer - r_inner)^3
    p.C = -E(r_outer) + 0.5 * (r_outer - r_inner) * E′(r_outer)
          -1.0 / 12.0 * (r_outer - r_inner)^2 * E′′(r_outer)

    return p
end

function ϕ(p::ZBL, r::Float64)
    return   0.18175 * p.e^(-3.1998*x) + 0.50986*p.e^(-0.94229*x) +
             0.28022*p.e^(-0.40290*x) + 0.2817*p.e^(-0.20162*x)
end

function zbl(p::ZBL, r::Float64)
    return 1 / (4.0 * π * p.ε0) * p.zi * p.zj * p.e^2 / r * ϕ(p, r/a)
end

"""
    S(r)

Switching function that ramps the energy, force, and curvature smoothly to zero
between an inner and outer cutoff. Here, the inner and outer cutoff are the same
for all pairs of atom types.
"""
function S(p::ZBL, r::Float64)
    if r < r_inner
        return p.C
    elseif r_inner < r && r < r_outer
        return p.A / 3.0 * (r - p.r_inner)^3 + p.B / 4.0 * (r - p.r_inner)^4 + p.C
    else
        println("Warning: ZBL calculation is incorrect (r > r_outer)")
        return 0.0
    end
end

function potential_energy(p::ZBL, r::Position, args...)
    return  zbl(p, norm(r)) + S(norm(r))
end

