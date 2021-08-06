include("../../src/PotentialLearning.jl")
using .PotentialLearning

path = "./"
params = get_conf_params(path)

# Get DFT data
dft_train_data, dft_val_data = get_dft_data(params)

# Get reference data
ref_train_data, ref_val_data = generate_data("ref", params)

# Get potential learning problem (e.g. A β = b)
snap = learning_problem(dft_train_data, ref_train_data, params)

# Solve potential learning problem (e.g. β = A \ b)
learn(snap, params)

# Validate potentials, forces, and stresses
metrics = error_metrics(snap, dft_val_data - ref_val_data, params)


