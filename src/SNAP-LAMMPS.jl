using LAMMPS
using LinearAlgebra:norm


"""
    Wrapper of the SNAP implementation of LAMMPS, built with LAMMPS.jl
    Mathematical formulation: A. P. Thompson et al.
                              http://dx.doi.org/10.1016/j.jcp.2014.12.018
"""
mutable struct SNAP_LAMMPS <: Potential
    β::Vector{Float64} # SNAP parameters to be fitted
    A::Matrix{Float64} # Matrix of potentials, forces, and stresses
    b::Vector{Float64} # = dft_training_data = potentials, forces, and stresses
    rows::Int64
    cols::Int64
    ncoeff::Int64
    no_atoms_per_conf::Int64
    no_atoms_per_type::Vector{Int64}
end

"""
    Creation of a SNAP_LAMMPS instance based on the configuration parameters
"""
function SNAP_LAMMPS(params::Dict)
    path = params["path"]
    ntypes = params["ntypes"]
    twojmax = params["twojmax"]
    no_atoms_per_type = params["no_atoms_per_type"]
    no_atomic_conf = params["no_atomic_conf"]
    rows = params["rows"]

    no_atoms_per_conf = sum(no_atoms_per_type)
    J = twojmax / 2.0
    ncoeff = round(Int, (J + 1) * (J + 2) * (( J + (1.5)) / 3. ) + 1)
    cols = 2 * ncoeff
    β = []
    A = Matrix{Float64}(undef, 0, 0)
    b = [] #b = dft_training_data
    p = SNAP_LAMMPS(β, A, b, rows, cols, ncoeff, no_atoms_per_conf, no_atoms_per_type)
    p.A = calc_A(path, p)
    return p
end

"""
    Error function to perform the learning process (Eq. 14)
    ToDO: make this function compatible with GalacticOptim.jl
"""
function error(β::Vector{Float64}, p, s::SNAP_LAMMPS)
    return norm(s.A * s.β - s.b)
end

"""
    Calculation of the A matrix of SNAP (Eq. 13)
"""
function calc_A(path::String, p::SNAP_LAMMPS)
    
    A = LMP(["-screen","none"]) do lmp

        A = Array{Float64}(undef, p.rows, p.cols) # bispectrum is 2*(ncoeff - 1) + 2

        for j in 1:p.rows
            open(string(path, "/GaN.commands")) do f 
                while !eof(f) 
                    line = replace(replace(readline(f), "\$PATH" => path),
                                   "\$ATOMICCONF" => string(j))
                    @info line
                    command(lmp, line)
                end
            end

            nlocal = extract_global(lmp, "nlocal")

            types = extract_atom(lmp, "type", LAMMPS.API.LAMMPS_INT)
            ids = extract_atom(lmp, "id", LAMMPS.API.LAMMPS_INT)

            @info "Progress" j nlocal

            ###
            # Energy
            ###

            bs = extract_compute(lmp, "SNA", LAMMPS.API.LMP_STYLE_ATOM,
                                             LAMMPS.API.LMP_TYPE_ARRAY)

            for i in length(p.no_atoms_per_type)
                @assert count(==(i), types) == p.no_atoms_per_type[i]
            end

            # Make bispectrum sum vector
            row = Float64[]

            for no_atoms in p.no_atoms_per_type
                push!(row, no_atoms)
                for k in 1:(p.ncoeff-1)
                    acc = 0.0
                    for n in 1:no_atoms
                        acc += bs[k, n]
                    end
                    push!(row, acc)
                end
            end

            A[j, :] = row

            command(lmp, "clear")
        end
        return A
    end
    return A
end

"""
    Calculation of the potential energy of a particular atomic configuration (j).
    This calculation requires accessing the SNAP implementation of LAMMPS.
"""
function potential_energy(params::Dict, j::Int64, p::Potential)
    # Calculate b
    path = params["path"]
    
    lmp = LMP(["-screen","none"])
    open(string(path, "/GaN.commands")) do f 
        while !eof(f) 
            line = replace(replace(readline(f), "\$PATH" => path),
                          "\$ATOMICCONF" => string(j))
            command(lmp, line)
        end
    end
    
    nlocal = extract_global(lmp, "nlocal")
    types = extract_atom(lmp, "type", LAMMPS.API.LAMMPS_INT)
    ids = extract_atom(lmp, "id", LAMMPS.API.LAMMPS_INT)
    bs = extract_compute(lmp, "SNA", LAMMPS.API.LMP_STYLE_ATOM,
                                     LAMMPS.API.LMP_TYPE_ARRAY)

    E_tot_acc = 0.0
    for (i, no_atoms) in enumerate(p.no_atoms_per_type)
        for n in 1:no_atoms
            E_atom_acc = p.β[p.ncoeff*(i-1)+1]
            for k in p.ncoeff*(i-1)+2:i*p.ncoeff
                k2 = k - p.ncoeff * (i - 1) - 1
                E_atom_acc += p.β[k] * bs[k2, n]
            end
            E_tot_acc += E_atom_acc
        end
    end
    
    command(lmp, "clear")
    return E_tot_acc
end

"""
    Calculation of the potential energy of a particular atomic configuration.
    It is based on the atomic positions of the configuration, the rcut, and a
    particular potential.
"""
function potential_energy(atomic_positions::Vector{Position}, rcut::Float64, p::Potential)
    acc = 0.0
    for i = 1:length(atomic_positions)
        for j = i:length(atomic_positions)
            r_diff = (atomic_positions[i] - atomic_positions[j])
            if norm(r_diff) <= rcut && norm(r_diff) > 0.0
                acc += potential_energy(i, j, r_diff, p)
            end
        end
    end
    return acc
end



