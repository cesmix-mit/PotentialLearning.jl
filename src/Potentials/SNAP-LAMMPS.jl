using LAMMPS
using LinearAlgebra:norm

"""
    SNAP_LAMMPS is a PotentialLearningProblem, it is based on the SNAP 
    implementation of LAMMPS, which is accessed through LAMMPS.jl.
    Mathematical formulation: A. P. Thompson et al. (10.1016/j.jcp.2014.12.018)
"""
mutable struct SNAP_LAMMPS <: PotentialLearningProblem
    path::String
    β::Vector{Float64}
    A::Matrix{Float64}            # Matrix of potentials, forces, and stresses
    b::Vector{Float64}            # DFT training data - Reference data
    dft_data::Vector{Float64}     # DFT training data
    ref_data::Vector{Float64}     # Reference data
    no_atoms_per_type::Vector
    ntypes::Int64
    no_train_atomic_conf::Int64
    no_atoms_per_conf::Int64
    twojmax::Int64
    ncoeff::Int64
    cols::Int64
    rcut::Float64
    fit_forces::Bool
end


"""
    SNAP_LAMMPS(dft_data::Vector{Float64}, ref_data::Vector{Float64}, params::Dict)

Creates a SNAP_LAMMPS.
"""
function SNAP_LAMMPS(dft_data::Vector{Float64}, ref_data::Vector{Float64}, params::Dict)
    path = params["global"]["path"]
    no_atoms_per_type = params["global"]["no_atoms_per_type"]
    ntypes = params["global"]["ntypes"]
    no_train_atomic_conf = params["global"]["no_train_atomic_conf"]
    no_atoms_per_conf = sum(no_atoms_per_type)
    no_atomic_conf = params["global"]["no_atomic_conf"]
    twojmax = params["global"]["twojmax"]
    rcut = params["global"]["rcut"]
    fit_forces = params["global"]["fit_forces"]
    
    J = twojmax / 2.0
    ncoeff = round(Int, (J + 1) * (J + 2) * (( J + (1.5)) / 3. ) + 1)
    cols = 2 * ncoeff
    β = []
    A = calc_A(path, rcut, twojmax, fit_forces, no_train_atomic_conf,
               no_atoms_per_conf, no_atoms_per_type, ncoeff, cols, ntypes)
    b = dft_data - ref_data
    
    return SNAP_LAMMPS( path, β, A, b, dft_data, ref_data, no_atoms_per_type,
                        ntypes, no_train_atomic_conf, no_atoms_per_conf, twojmax,
                        ncoeff, cols, rcut, fit_forces)
end

"""
    error(β::Vector{Float64}, s::SNAP_LAMMPS)

Error function needed to perform the learning process.
"""
function error(β::Vector{Float64}, s::SNAP_LAMMPS)
    return norm(s.A * β - s.b)
end

"""
    get_bispectrums(lmp, path::String, rcut::Float64, twojmax::Int64)

Calcuulates the bispectrums components and its derivatives.
"""
function get_bispectrums(lmp, path::String, rcut::Float64, twojmax::Int64)
    read_data_str = "read_data " * path

    command(lmp, "log none")
    command(lmp, "units metal")
    command(lmp, "boundary p p p")
    command(lmp, "atom_style atomic")
    command(lmp, "atom_modify map array")
    command(lmp, read_data_str)
    command(lmp, "pair_style zero $rcut")
    command(lmp, "pair_coeff * *")
    command(lmp, "compute PE all pe")
    command(lmp, "compute S all pressure thermo_temp")
    command(lmp, "compute SNA all sna/atom $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "compute SNAD all snad/atom $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "compute SNAV all snav/atom $rcut 0.99363 $twojmax 0.5 0.5 1.0 0.5 rmin0 0.0 bzeroflag 0 quadraticflag 0 switchflag 1")
    command(lmp, "thermo_style custom pe")
    command(lmp, "run 0")

    ## Extract bispectrum
    bs = extract_compute(lmp, "SNA",  LAMMPS.API.LMP_STYLE_ATOM,
                                      LAMMPS.API.LMP_TYPE_ARRAY)
    deriv_bs = extract_compute(lmp, "SNAD", LAMMPS.API.LMP_STYLE_ATOM,
                                            LAMMPS.API.LMP_TYPE_ARRAY)
    return bs, deriv_bs
end

"""
    calc_A(path::String, rcut::Float64, twojmax::Int64, fit_forces::Bool,
           no_train_atomic_conf::Int64, no_atoms_per_conf::Int64,
           no_atoms_per_type::Vector, ncoeff::Int64, cols::Int64, ntypes::Int64)

Calculates the matrix `A` (See Eq. 13, 10.1016/j.jcp.2014.12.018).
This calculation requires accessing the SNAP implementation of LAMMPS.
"""
function calc_A(path::String, rcut::Float64, twojmax::Int64, fit_forces::Bool,
                no_train_atomic_conf::Int64, no_atoms_per_conf::Int64,
                no_atoms_per_type::Vector, ncoeff::Int64, cols::Int64, ntypes::Int64)
    
    A = LMP(["-screen","none"]) do lmp

        A_potential = Array{Float64}(undef, no_train_atomic_conf, cols)
        
        A_forces = fit_forces ? Array{Float64}(undef, 3 * no_atoms_per_conf *
                                no_train_atomic_conf, cols) : Array{Float64}(undef, 0, cols)
        
        for j in 1:no_train_atomic_conf
            data = joinpath(path, "DATA", string(j), "DATA")
            bs, deriv_bs = get_bispectrums(lmp, data, rcut, twojmax)
            
            # Create a row of the potential energy block
            row = Vector{Float64}()
            for no_atoms in no_atoms_per_type
                push!(row, no_atoms)
                for k in 1:(ncoeff-1)
                    acc = 0.0
                    for n in 1:no_atoms
                        acc += bs[k, n]
                    end
                    push!(row, acc)
                end
            end
            A_potential[j, :] = row
            
            if fit_forces
                # Create a set of rows of the force block
                k = (j - 1) * 3 * no_atoms_per_conf + 1 
                for n = 1:no_atoms_per_conf
                    for c = [1, 2, 3] # component x, y, z
                        row = Vector{Float64}()
                        for t = 1:ntypes # e.g. 2 : Ga and N
                            offset = (t-1) * (ncoeff-1) + (c-1) * ntypes * (ncoeff-1)
                            row = [row; [0.0; deriv_bs[1 + offset:(ncoeff-1) + offset, n]]]
                        end
                        A_forces[k, :] = row
                        k += 1
                    end
                end
            end
            
            command(lmp, "clear")
        end
        return [A_potential; A_forces]
    end
    
    return A
end


"""
    potential_energy(p::Potential, j::Int64)

Calculates the potential energy of a particular atomic configuration (j)
using the fitted parameters β.
This calculation requires accessing the SNAP implementation of LAMMPS.
"""
function potential_energy(p::PotentialLearningProblem, j::Int64)

    data = joinpath(p.path, "DATA", string(j), "DATA")
    lmp = LMP(["-screen","none"])
    bs, deriv_bs = get_bispectrums(lmp, data, p.rcut, p.twojmax)
    
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
    forces(p::Potential, j::Int64)

Calculates the forces of a particular atomic configuration (j) using the 
fitted parameters β. 
This calculation requires accessing the SNAP implementation of LAMMPS.
"""
function forces(p::PotentialLearningProblem, j::Int64)

    data = joinpath(p.path, "DATA", string(j), "DATA")
    lmp = LMP(["-screen","none"])
    bs, deriv_bs = get_bispectrums(lmp, data, p.rcut, p.twojmax)
    
    forces = Vector{Force}()
    
    for n = 1:p.no_atoms_per_conf
        for t = 1:p.ntypes # e.g. 2 : Ga and N
            f = [0.0, 0.0, 0.0]
            for c = [1, 2, 3] # component x, y, z
                offset = (t-1) * (p.ncoeff-1) + (c-1) * p.ntypes * (p.ncoeff-1)
                deriv_bs_row = [0.0; deriv_bs[1 + offset:(p.ncoeff-1) + offset, n]]
                f[c] = sum([p.β[k + (t-1) * p.ncoeff] * deriv_bs_row[k]
                            for k in 1:length(deriv_bs_row)])
            end
            push!(forces, Force(f))
        end
    end
    
    command(lmp, "clear")
    return forces
end



