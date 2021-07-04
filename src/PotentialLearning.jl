
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
include("InputLoading.jl")
include("EmpiricalPotentials.jl")
include("SNAP-LAMMPS.jl")

using GalacticOptim, Optim, Printf

function fit(p)
    #β0 = zeros(p.rows)
    #prob = OptimizationProblem(error, β0, [], p)
    #p.β = solve(prob, NelderMead())
    p.β = p.A \ p.b
end

function run(path)
    # Currently this function is hardcoded to run a SNAP-LAMMPS example

    # Load atomic configurations ###############################################
    params = load_conf_params(path)
    
    # Get DFT data #############################################################
    dft_data = load_dft_data(path, params["DFT_model"])
    
    # Calculate potential energy per configuration (vector b)
    potential_energy_per_conf = 
            [potential_energy(params["positions_per_conf"][j], params["rcut"],
                              params["no_atoms_per_conf"], dft_data) 
             for j = 1:params["rows"]]
    
    # Create potential to fit ##################################################
    p = SNAP_LAMMPS(path, params, potential_energy_per_conf)
    
    # Potential Learning #######################################################
    fit(p)
    
    # Check Potential ##########################################################
    @show norm(p.A * p.β - p.b)
#    @printf("Potential Energy, Fitted Potential Energy, Error (%%)\n")
#    for j = params["rows"]+1:params["no_atomic_conf"]
#        E_tot = potential_energy(params["positions_per_conf"][j],
#                                 params["rcut"], params["no_atoms_per_conf"], dft_data)
#        E_tot_fit = potential_energy(path, p.β, p.ncoeff, pparams["no_atoms_per_conf"])
#        @printf("%0.2f, %0.2f, %0.2f\n", E_tot, E_tot_fit,
#                abs(E_tot - E_tot_fit) / E_tot * 100.)
#    end
    
    # Calculate force ##########################################################
    # f = p.get_forces()
    
    return p
end

end

using .PotentialLearning
PotentialLearning.run("../examples/GaNData/")
