using Test

@testset "GaN-SNAP-LAMMPS" begin

    params = get_conf_params("../examples/GaN-SNAP-LAMMPS/")

    # Get DFT data
    dft_train_data, dft_val_data = generate_data("dft", params)

    # Get reference data
    ref_train_data, ref_val_data = generate_data("ref", params)

    # Get potential learning problem (e.g. A β = b)
    snap = learning_problem(dft_train_data, ref_train_data, params)

    # Solve potential learning problem (e.g. β = A \ b)
    learn(snap, params)

    # Validate potentials, forces, and stresses
    rel_error = validate(snap, dft_val_data - ref_val_data, params)

    @test rel_error < 0.1

end

