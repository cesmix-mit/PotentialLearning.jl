using Test

@testset "GaN-SNAP-LAMMPS" begin

    # Load configuration parameters (e.g. potential, solver, DFT model)
    path = "../examples/GaN-SNAP-LAMMPS/"
    params = load_conf_params(path)

    # Load DFT data
    dft_training_data, dft_validation_data = load_dft_data(params)

    # Define potential learning problem (e.g. SNAP linear system)
    snap = SNAP_LAMMPS(dft_training_data, params)

    # Learn potentials, forces, and stresses (e.g. calculate β of the system A β = b)
    learn(snap, params)

    # Validate potentials, forces, and stresses
    rel_error = validate_potentials(snap, dft_validation_data, params)
    @test rel_error < 0.1

end

