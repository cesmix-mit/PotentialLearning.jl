using LAMMPS
using LinearAlgebra:norm


"""
    SNAP_LAMMPS is a PotentialLearningProblem. It can be solved by 
    It is based on the SNAP implementation of LAMMPS, which is accessed through LAMMPS.jl.
    Mathematical formulation: A. P. Thompson et al. (10.1016/j.jcp.2014.12.018)
"""
mutable struct SNAP_LAMMPS <: PotentialLearningProblem
    β::Vector{Float64} # SNAP parameters to be fitted
    A::Matrix{Float64} # Matrix of potentials, forces, and stresses
    b::Vector{Float64} # = dft_training_data = potentials, forces, and stresses
    no_train_atomic_conf::Int64
    cols::Int64
    ntypes::Int64
    ncoeff::Int64
    no_atoms_per_conf::Int64
    no_atoms_per_type::Vector{Int64}
end

"""
    SNAP_LAMMPS(dft_training_data::Vector{Float64}, params::Dict)

Creation of a SNAP_LAMMPS instance based on the DFT training data and the 
configuration parameters.
"""
function SNAP_LAMMPS(dft_training_data::Vector{Float64}, params::Dict)
    path = params["path"]
    ntypes = params["ntypes"]
    rcut = params["rcut"]
    twojmax = params["twojmax"]
    no_atoms_per_type = params["no_atoms_per_type"]
    no_atomic_conf = params["no_atomic_conf"]
    no_train_atomic_conf = params["no_train_atomic_conf"]

    no_atoms_per_conf = sum(no_atoms_per_type)
    J = twojmax / 2.0
    ncoeff = round(Int, (J + 1) * (J + 2) * (( J + (1.5)) / 3. ) + 1)
    cols = 2 * ncoeff
    β = []
    A = Matrix{Float64}(undef, 0, 0)
    
#    # Load reference energy, forces and stresses
#    ref_model = Symbol(params["Reference_model"])
#    ref_data = @eval $ref_model($params)
#    
#    potential_ref_data = [potential_energy(ref_potential, positions_per_conf[j], rcut)
#                          for j = 1:no_train_atomic_conf]
#    force_ref_data = [forces(ref_potential, positions_per_conf[j], rcut)
#                      for j = 1:no_train_atomic_conf]
#    force_ref_data_lin = Vector{Float64}()
#    for i = 1:length(force_ref_data), j = 1:length(force_ref_data[i]), k = 1:3
#        push!(force_ref_data_lin, force_ref_data[i][j][k])
#    end
#    ref_training_data = [potential_ref_data; force_ref_data_lin]
    ref_training_data = zeros(length(dft_training_data))

    b = dft_training_data + ref_training_data
    p = SNAP_LAMMPS(β, A, b, no_train_atomic_conf, cols, ntypes, ncoeff, no_atoms_per_conf, no_atoms_per_type)
    p.A = calc_A(path, rcut, twojmax, p)
    return p
end

"""
    error(β::Vector{Float64}, s::SNAP_LAMMPS)

Error function to perform the learning process (Eq. 14, 10.1016/j.jcp.2014.12.018)
"""
function error(β::Vector{Float64}, s::SNAP_LAMMPS)
    return norm(s.A * β - s.b)
end

"""
    run_snap(lmp, path, rcut, twojmax)

Execution of SNAP, LAMMPS to extract bispectrum components
"""
function run_snap(lmp, path::String, rcut::Float64, twojmax::Int64)
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
    calc_A(path::String, p::SNAP_LAMMPS)

Calculation of the A matrix of SNAP (Eq. 13, 10.1016/j.jcp.2014.12.018)
"""
function calc_A(path::String, rcut::Float64, twojmax::Int64, p::SNAP_LAMMPS)
    
    A = LMP(["-screen","none"]) do lmp

        A_potential = Array{Float64}(undef, p.no_train_atomic_conf, p.cols)
        A_forces    = Array{Float64}(undef, 3 * p.no_atoms_per_conf *
                                            p.no_train_atomic_conf, p.cols)
        
        for j in 1:p.no_train_atomic_conf
            data = joinpath(path, "DATA", string(j), "DATA")
            bs, deriv_bs = run_snap(lmp, data, rcut, twojmax)

            # Create a row of the potential energy block
            row = Vector{Float64}()
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
            A_potential[j, :] = row
            
            # Create a set of rows of the force block
            k = (j - 1) * 3 * p.no_atoms_per_conf + 1 
            for n = 1:p.no_atoms_per_conf
                for c = [1, 2, 3] # component x, y, z
                    row = Vector{Float64}()
                    for t = 1:p.ntypes # e.g. 2 : Ga and N
                        offset = (t-1) * (p.ncoeff-1) + (c-1) * p.ntypes * (p.ncoeff-1)
                        row = [row; [0.0; deriv_bs[1 + offset:(p.ncoeff-1) + offset, n]]]
                    end
                    A_forces[k, :] = row
                    k += 1
                end
            end

            command(lmp, "clear")
        end
        return [A_potential; A_forces]
    end
    
    return A
end

"""
    potential_energy(p::PotentialLearningProblem, j::Int64, params::Dict)

Calculation of the potential energy of a particular atomic configuration (j).
This calculation requires accessing the SNAP implementation of LAMMPS.
"""
function potential_energy(p::PotentialLearningProblem, j::Int64, params::Dict)
    # Calculate b
    path = params["path"]
    rcut = params["rcut"]
    twojmax = params["twojmax"]
    
    data = joinpath(path, "DATA", string(j), "DATA")
    lmp = LMP(["-screen","none"])
    bs, deriv_bs = run_snap(lmp, data, rcut, twojmax)
    
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



