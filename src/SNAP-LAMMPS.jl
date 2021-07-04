using LAMMPS

mutable struct SNAP_LAMMPS
    β::Vector{Float64}
    A::Matrix{Float64}
    b::Vector{Float64}
    rows::Int64
    cols::Int64
    ncoeff::Int64
    no_atoms_per_conf::Int64
    no_atoms_per_type::Vector{Int64}
    
    SNAP_LAMMPS(path, ntypes, twojmax, no_atoms_per_type, no_atomic_conf, rows, potential_energy_per_conf) = 
    begin
        no_atoms_per_conf = sum(no_atoms_per_type)
        J = twojmax / 2.0
        ncoeff = round(Int, (J + 1) * (J + 2) * (( J + (1.5)) / 3. ) + 1)
        cols = 2 * ncoeff
        β = []
        A = Matrix{Float64}(undef, 0, 0)
        b = potential_energy_per_conf
        p = new(β, A, b, rows, cols, ncoeff, no_atoms_per_conf, no_atoms_per_type)
        p.A = calc_A(path, p)
        return p
    end
end

error(β::Vector{Float64}, p, s::SNAP_LAMMPS) = norm(s.A * s.β - s.b)

function calc_A(path::String, p::SNAP_LAMMPS)
    
    A = LMP(["-screen","none"]) do lmp

        A = Array{Float64}(undef, p.rows, p.cols) # bispectrum is 2*(ncoeff - 1) + 2

        for m in 1:p.rows
            read_data_str = "read_data " * joinpath(path, string(m), "DATA")

            # ToDo: This needs to be in a conf file
            command(lmp, "units metal")
            command(lmp, "boundary p p p")
            command(lmp, "atom_style atomic")
            command(lmp, "atom_modify map array")
            command(lmp, read_data_str)
            command(lmp, "pair_style snap")
            command(lmp, "pair_coeff * * GaN.snapcoeff GaN.snapparam Ga N")
            command(lmp, "compute PE all pe")
            command(lmp, "compute S all pressure thermo_temp")
            command(lmp, "compute SNA all sna/atom 3.5 0.99363 6 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
            command(lmp, "compute SNAD all snad/atom 3.5 0.99363 6 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
            command(lmp, "compute SNAV all snav/atom 3.5 0.99363 6 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
            command(lmp, "thermo_style custom pe")
            command(lmp, "dump 2 all custom 100 dump.forces fx fy fz")
            command(lmp, "run 0")

            nlocal = extract_global(lmp, "nlocal")

            types = extract_atom(lmp, "type", LAMMPS.API.LAMMPS_INT)
            ids = extract_atom(lmp, "id", LAMMPS.API.LAMMPS_INT)

            @info "Progress" m nlocal

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

            A[m, :] = row

            command(lmp, "clear")
        end
        return A
    end
    return A
end

function potential_energy(path::String, p::SNAP_LAMMPS)
    ## Calculate b
    lmp = LMP(["-screen","none"]) 
    read_data_str = string("read_data ", path)
    
    # ToDo: This needs to be in a conf file
    command(lmp, "units metal")
    command(lmp, "boundary p p p")
    command(lmp, "atom_style atomic")
    command(lmp, "atom_modify map array")
    command(lmp, read_data_str)
    command(lmp, "pair_style snap")
    command(lmp, "pair_coeff * * GaN.snapcoeff GaN.snapparam Ga N")
    command(lmp, "compute PE all pe")
    command(lmp, "compute S all pressure thermo_temp")
    command(lmp, "compute SNA all sna/atom 3.5 0.99363 6 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "compute SNAD all snad/atom 3.5 0.99363 6 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "compute SNAV all snav/atom 3.5 0.99363 6 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "thermo_style custom pe")
    command(lmp, "dump 2 all custom 100 dump.forces fx fy fz")
    command(lmp, "run 0")
    nlocal = extract_global(lmp, "nlocal")
    types = extract_atom(lmp, "type", LAMMPS.API.LAMMPS_INT)
    ids = extract_atom(lmp, "id", LAMMPS.API.LAMMPS_INT)
    bs = extract_compute(lmp, "SNA", LAMMPS.API.LMP_STYLE_ATOM,
                                     LAMMPS.API.LMP_TYPE_ARRAY)
                                     
                                     
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

    E_tot_acc = 0.0
    for (i, no_atoms) in enumerate(p.no_atoms_per_type)
        for n in 1:no_atoms
            E_atom_acc = β[1]
            for k in ncoeff*(i-1)+2:i*ncoeff
                k2 = k - 1
                E_atom_acc += β[k] * bs[k2, n]
            end
            E_tot_acc += E_atom_acc
        end
    end
    
    command(lmp, "clear")
    return E_tot_acc
end

# Calculates the potential energy of a particular atomic configuration
function potential_energy(rs, rcut, no_atoms_per_conf, p)
    acc = 0.0
    for i = 1:no_atoms_per_conf
        for j = i:no_atoms_per_conf
            r_diff = rs[i] - rs[j]
            if norm(r_diff) <= rcut && norm(r_diff) > 0.0
                acc += potential_energy(i, j, r_diff, p)
            end  
        end
    end
    return acc
end



