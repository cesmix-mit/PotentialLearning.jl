
################################################################################
#
#    Module PotentialLearning.jl
#
#    How to use it:
#        julia> include("PotentialLearning.jl")
#        julia> using .PotentialLearning
#        julia> PotentialLearning.run()
#
################################################################################

module PotentialLearning

include("Utils.jl")
include("AtomicConfigurations.jl")
include("EmpiricalPotentials.jl")
include("SNAP-LAMMPS.jl")

using GalacticOptim, Optim, Printf

function fit(p)
    #β0 = zeros(p.rows)
    #prob = OptimizationProblem(error, β0, [], p)
    #p.β = solve(prob, NelderMead())
    p.β = p.A \ p.b
end

#function load_conf_params(path)
#    no_Ga = 96
#    no_N = 96
#    no_atoms_per_type = [no_Ga, no_N]
#    no_atoms_per_conf = sum(no_atoms_per_type)
#    no_atomic_conf = 61
#    positions_per_conf = load_positions_per_conf(path, no_atoms_per_conf, no_atomic_conf)
#    rcut = 3.5
#    ntypes = 2
#    twojmax = 5
#    rows = 48
#end

function run(path)
    # Currently this function is hardcoded to run a SNAP-LAMMPS example

    # Load atomic configurations ###############################################
    #params = load_conf_params(path)
    no_Ga = 96
    no_N = 96
    no_atoms_per_type = [no_Ga, no_N]
    no_atoms_per_conf = sum(no_atoms_per_type)
    no_atomic_conf = 61
    positions_per_conf = load_positions_per_conf(path, no_atoms_per_conf, no_atomic_conf)
    rcut = 3.5
    ntypes = 2
    twojmax = 5
    rows = 48
    
    # Get DFT data #############################################################
    #dft_data = load_dft_data(path, params["dft_model"])
    dft_data = load_GaN(path)
    
    # Calculate potential energy per configuration (vector b)
    potential_energy_per_conf = 
            [potential_energy(positions_per_conf[i], rcut, no_atoms_per_conf, dft_data) 
             for i = 1:rows]
    
    # Create potential to fit ##################################################
    p = SNAP_LAMMPS(path, ntypes, twojmax, no_atoms_per_type, no_atomic_conf,
                    rows, potential_energy_per_conf)
    
    # Potential Learning #######################################################
    fit(p)
    
    # Check Potential ##########################################################
    @show norm(p.A * p.β - p.b)
    @printf("Potential Energy, Fitted Potential Energy, Error (%%)\n")
    for j = no_Ga+1:no_atomic_conf
        E_tot = potential_energy(positions_per_conf[j], rcut, no_atoms_per_conf, gan)
        E_tot_fit = potential_energy(path, β, ncoeff, N1, N)
        @printf("%0.2f, %0.2f, %0.2f\n", E_tot, E_tot_fit,
                abs(E_tot - E_tot_fit) / E_tot * 100.)
    end
    
    # Calculate force ##########################################################
    # f = p.get_forces()
    
    return p
end

end

using .PotentialLearning
PotentialLearning.run("../examples/GaNData/")
