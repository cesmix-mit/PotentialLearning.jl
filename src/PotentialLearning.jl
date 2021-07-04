
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

function run()
    # Currently this function is hardcoded to run a SNAP-LAMMPS example

    # Load atomic configurations ###############################################
    path = "GaNData/"
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
    
    # Load DFT surrogate potential: GaN potential
    ε_Ga_Ga = 0.643; σ_Ga_Ga = 2.390
    ε_N_N = 1.474; σ_N_N = 1.981
    A_Ga_N = 608.54; ρ_Ga_N = 0.435
    q_Ga = 3.0; q_N = -3.0; ε0 = 55.26349406 # e2⋅GeV−1⋅fm−1 ToDo: check this
    lj_Ga_Ga = LennardJones(ε_Ga_Ga, σ_Ga_Ga)
    lj_N_N = LennardJones(ε_N_N, σ_N_N)
    bm_Ga_N = BornMayer(A_Ga_N, ρ_Ga_N)
    c = Coulomb(q_Ga, q_N, ε0)
    gan = GaN(lj_Ga_Ga, lj_N_N, bm_Ga_N, c, no_Ga, no_N)
    
    # Calculate potential energy per configuration (vector b)
    potential_energy_per_conf = 
            [potential_energy(positions_per_conf[i], rcut, no_atoms_per_conf, gan) 
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
    #f = p.get_forces()
    
    return p
end

end

using .PotentialLearning
PotentialLearning.run()
