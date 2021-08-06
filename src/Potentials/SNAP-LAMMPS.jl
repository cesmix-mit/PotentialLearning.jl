using LAMMPS
using LinearAlgebra:norm

"""
    SNAP_LAMMPS is a PotentialLearningProblem, it is based on the SNAP 
    implementation of LAMMPS, which is accessed through LAMMPS.jl.
    Mathematical formulation: A. P. Thompson et al. (10.1016/j.jcp.2014.12.018)
"""
mutable struct SNAP_LAMMPS <: PotentialLearningProblem

    path::String                      # Path of the example folder
    β::Vector{Float64}                # Fitted parameters
    A::Matrix{Float64}                # Matrix of potentials, forces, and stresses.
    b::Vector{Float64}                # DFT training data - Reference data
    dft_data::Vector{Float64}         # DFT training data
    ref_data::Vector{Float64}         # Reference data
    no_atoms_per_type::Vector         # No. of atoms per type
    no_atoms_per_conf::Int64          # No. of atoms per atomic configurations
    no_atomic_conf::Int64             # No. of atomic configurations
    no_train_atomic_conf::Int64       # No. of trainning atomic configurations
    rcutfac::Float64                  # Scale factor applied to all cutoff radii (positive real)
    rfac0::Float64                    # Parameter in distance to angle conversion (0 < rcutfac < 1)
    twojmax::Int64                    # SNAP-LAMMPS parameter
    cutoff_radii::Vector{Float64}     # List of cutoff radii, one for each type
    neighbor_weights::Vector{Float64} # List of neighbor weights, one for each type
    ncoeff::Int64                     # ncoeff-1 = no. of param. associated to an atom type
    cols::Int64                       # No. of columns of matrix `A`
    numTypes::Int64                   # No. of atom types
    opt_pars::Dict                    # Optional parameters: rmin0, switchflag,
                                      # bzeroflag, quadraticflag, chem, chemflag,
                                      # bnormflag, wselfallflag, switchflag
    fit_forces::Bool                  # Enable/disable force fitting

end


"""
    SNAP_LAMMPS(dft_data::Vector{Float64}, ref_data::Vector{Float64}, params::Dict)

Creates a SNAP_LAMMPS.
"""
function SNAP_LAMMPS(dft_data::Vector{Float64}, ref_data::Vector{Float64}, params::Dict)

    β = []
    A = Matrix{Float64}(undef, 0, 0)
    b = dft_data - ref_data
    pars = params["global"]
    path = pars["path"]
    aux = pars["no_atoms_per_type"]
    no_atoms_per_type = length(aux) > 1 ? aux : [aux]
    no_atoms_per_conf = sum(no_atoms_per_type)
    no_atomic_conf = pars["no_atomic_conf"]
    no_train_atomic_conf = pars["no_train_atomic_conf"]
    rcutfac = pars["rcutfac"]
    rfac0 = pars["rfac0"]
    twojmax = pars["twojmax"]
    J = twojmax / 2.0
    ncoeff = round(Int, (J + 1) * (J + 2) * (( J + (1.5)) / 3. ) + 1)
    cols = 2 * ncoeff
    aux = pars["cutoff_radii"]
    cutoff_radii = length(aux) > 1 ? aux : [aux]
    aux = pars["neighbor_weights"]
    neighbor_weights = length(aux) > 1 ? aux : [aux]
    numTypes = pars["numTypes"]

    opt_pars = Dict()
    if haskey(pars, "rmin0") opt_pars["rmin0"] = pars["rmin0"] end
    if haskey(pars, "switchflag") opt_pars["switchflag"] = pars["switchflag"] end
    if haskey(pars, "bzeroflag") opt_pars["bzeroflag"] = pars["bzeroflag"] end
    if haskey(pars, "quadraticflag") opt_pars["quadraticflag"] = pars["quadraticflag"] end
    if haskey(pars, "chem") opt_pars["chem"] = pars["chem"] end
    if haskey(pars, "chemflag") opt_pars["chemflag"] = pars["chemflag"] end
    if haskey(pars, "bnormflag") opt_pars["bnormflag"] = pars["bnormflag"] end
    if haskey(pars, "wselfallflag") opt_pars["wselfallflag"] = pars["wselfallflag"] end
    if haskey(pars, "switchflag") opt_pars["switchflag"] = pars["switchflag"] end
    
    fit_forces = pars["fit_forces"]
    
    p = SNAP_LAMMPS( path, β, A, b, dft_data, ref_data, no_atoms_per_type, no_atoms_per_conf,
                     no_atomic_conf, no_train_atomic_conf, rcutfac, rfac0, twojmax,
                     cutoff_radii, neighbor_weights, ncoeff, cols, numTypes, opt_pars,
                     fit_forces)
                     
    p.A = calc_A(p)
    
    return p
end

"""
    error(β::Vector{Float64}, s::SNAP_LAMMPS)

Error function needed to perform the learning process.
"""
function error(β::Vector{Float64}, s::SNAP_LAMMPS)
    return norm(s.A * β - s.b)
end

"""
    get_bispectrums(lmp, data_path::String, p::SNAP_LAMMPS)

Calculates the bispectrums components and its derivatives.
See https://docs.lammps.org/compute_sna_atom.html
"""
function get_bispectrums(lmp, data_path::String, p::SNAP_LAMMPS)

    args = ""
    # scale factor applied to all cutoff radii (positive real)
    args *= "$(p.rcutfac) "
    # parameter in distance to angle conversion (0 < rcutfac < 1)
    args *= "$(p.rfac0) "
    # band limit for bispectrum components (non-negative integer)
    args *= "$(p.twojmax) "
    # R_1, R_2,… = list of cutoff radii, one for each type (distance units)
    args *= join(["$r " for r in p.cutoff_radii])
    # w_1, w_2,… = list of neighbor weights, one for each type
    args *= join(["$r " for r in p.neighbor_weights])
    # zero or more keyword/value pairs may be appended
    # keyword = rmin0 or switchflag or bzeroflag or quadraticflag or chem or bnormflag or wselfallflag
    args *= haskey(p.opt_pars, "rmin0") ? "rmin0 $(p.opt_pars["rmin0"]) " : ""
    args *= haskey(p.opt_pars, "switchflag") ? "switchflag $(p.opt_pars["switchflag"]) " : ""
    args *= haskey(p.opt_pars, "bzeroflag") ? "bzeroflag $(p.opt_pars["bzeroflag"]) " : ""
    args *= haskey(p.opt_pars, "quadraticflag") ? "quadraticflag $(p.opt_pars["quadraticflag"]) " : ""
    args *= haskey(p.opt_pars, "chem") ? "chem " * join(["$r " for r in p.opt_pars["chem"]]) * " " : ""
    args *= haskey(p.opt_pars, "chemflag") ? "chemflag $(p.opt_pars["chemflag"]) " : ""
    args *= haskey(p.opt_pars, "bnormflag") ? "bnormflag $(p.opt_pars["bnormflag"]) " : ""
    args *= haskey(p.opt_pars, "wselfallflag") ? "wselfallflag $(p.opt_pars["wselfallflag"]) " : ""
    args *= haskey(p.opt_pars, "switchflag") ? "switchflag $(p.opt_pars["switchflag"]) " : ""

    read_data_str = "read_data " * data_path
    command(lmp, "log none")
    command(lmp, "units metal")
    command(lmp, "boundary p p p")
    command(lmp, "atom_style atomic")
    command(lmp, "atom_modify map array")
    command(lmp, read_data_str)
#    command(lmp, "pair_style hybrid/overlay zero 10.0 zbl 4.0 4.8")
#    command(lmp, "pair_coeff * * zero")
#    command(lmp, "pair_coeff * * zbl 73 73")
    command(lmp, "pair_style zero $(p.rcutfac)")
    command(lmp, "pair_coeff * *")
    command(lmp, "compute PE all pe")
    command(lmp, "compute S all pressure thermo_temp")
    command(lmp, "compute SNA all sna/atom " * args)
    command(lmp, "compute SNAD all snad/atom " * args)
    command(lmp, "compute SNAV all snav/atom " * args)
    command(lmp, "thermo_style custom pe")
    command(lmp, "run 0")

    # Extract bispectrum
    bs = extract_compute(lmp, "SNA",  LAMMPS.API.LMP_STYLE_ATOM,
                                      LAMMPS.API.LMP_TYPE_ARRAY)
    deriv_bs = extract_compute(lmp, "SNAD", LAMMPS.API.LMP_STYLE_ATOM,
                                            LAMMPS.API.LMP_TYPE_ARRAY)
    return bs, deriv_bs
end

"""
    calc_A(params::Dict, p::SNAP_LAMMPS)

Calculates the matrix `A` (See Eq. 13, 10.1016/j.jcp.2014.12.018).
This calculation requires accessing the SNAP implementation of LAMMPS.
"""
function calc_A(p::SNAP_LAMMPS)
    
    A = LMP(["-screen","none"]) do lmp

        A_potential = Array{Float64}(undef, p.no_train_atomic_conf, p.cols)
        
        A_forces = p.fit_forces ? Array{Float64}(undef, 3 * p.no_atoms_per_conf *
                                  p.no_train_atomic_conf, p.cols) : Array{Float64}(undef, 0, p.cols)
        
        for j in 1:p.no_train_atomic_conf
            data_path = joinpath(p.path, "DATA", string(j), "DATA")
            bs, deriv_bs = get_bispectrums(lmp, data_path, p)
            
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
            
            if p.fit_forces
                # Create a set of rows of the force block
                k = (j - 1) * 3 * p.no_atoms_per_conf + 1 
                for n = 1:p.no_atoms_per_conf
                    for c = [1, 2, 3] # component x, y, z
                        row = Vector{Float64}()
                        for t = 1:p.numTypes # e.g. 2 : Ga and N
                            a = 1 + (p.ncoeff-1) * (c-1) + 3 * (p.ncoeff-1) * (t-1)
                            b = a + p.ncoeff - 2
                            row = [row; [0.0; deriv_bs[a:b, n]]]
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

Calculates the potential energy of a particular atomic configuration (j) using
the fitted parameters β. 
This calculation requires accessing the SNAP implementation of LAMMPS.
"""
function potential_energy(p::PotentialLearningProblem, j::Int64)

    data_path = joinpath(p.path, "DATA", string(j), "DATA")
    lmp = LMP(["-screen","none"])
    bs, deriv_bs = get_bispectrums(lmp, data_path, p)
    
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
See https://docs.lammps.org/compute_sna_atom.html#compute-snad-atom-command
"""
function forces(p::PotentialLearningProblem, j::Int64)

    data_path = joinpath(p.path, "DATA", string(j), "DATA")
    lmp = LMP(["-screen","none"])
    bs, deriv_bs = get_bispectrums(lmp, data_path, p)
    
    forces = Vector{Force}()

    for n = 1:p.no_atoms_per_conf
        f = [0.0, 0.0, 0.0]
        for c = [1, 2, 3] # component x, y, z
            deriv_bs_row = []
            for t = 1:p.numTypes
                a = 1 + (p.ncoeff-1) * (c-1) + 3 * (p.ncoeff-1) * (t-1)
                b = a + p.ncoeff - 2
                deriv_bs_row = [deriv_bs_row; [0.0; deriv_bs[a:b, n]]]
            end
            f[c] = sum(deriv_bs_row .* p.β)
        end
        push!(forces, Force(f))
    end
    
    command(lmp, "clear")
    return forces
end



