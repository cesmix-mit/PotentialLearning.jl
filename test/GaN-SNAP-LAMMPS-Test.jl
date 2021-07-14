using Test

include("../src/Utils.jl")
include("../src/SNAP-LAMMPS.jl")

@testset "GaN-SNAP-LAMMPS" begin

    # Load learning parameters
    path = "../examples/GaN-SNAP-LAMMPS/"
    learning_params = load_learning_params(path)
    
    # Load DFT data
    dft_training_data, dft_validation_data = load_dft_data(learning_params)
    
    # Load potential
    snap = SNAP_LAMMPS(learning_params)
    
    # Learn potential, forces, and stresses
    learn(snap, dft_training_data, learning_params)
    
    # Validate potential, forces, and stresses
    rel_error = validate(snap, dft_validation_data, learning_params)
    @test rel_error < 0.1

end

