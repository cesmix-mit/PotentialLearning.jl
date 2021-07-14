using Test

@testset "GaN-SNAP-LAMMPS" begin

    # Load configuration parameters
    path = "../examples/GaN-SNAP-LAMMPS/"
    params = load_conf_params(path)
    
    # Load DFT data
    dft_training_data, dft_validation_data = load_dft_data(params)
    
    # Load potential
    snap = SNAP_LAMMPS(params)
    
    # Learn potentials, forces, and stresses
    learn(snap, dft_training_data, params)
    
    # Validate potentials, forces, and stresses
    rel_error = validate(snap, dft_validation_data, params)
    @test rel_error < 0.1

end

