# [WIP] PotentialLearning.jl: The Julia Library of Molecular Dynamics Potentials

An Open Source library for active training and fast calculation of molecular dynamics potentials for atomistic simulations of materials. 

## Features under development
- Surrogate DFT data generation
  - Gallium nitride model
- Integration with GalacticOptim.jl to perform the optimization process
- Integration with LAMMPS.jl to access the SNAP implementation of LAMMPS
- Implementation of a pure Julia version of SNAP
  - GPU implementation using KernelAbstractions.jl

## Installation instructions

To install PotentialLearning.jl in Julia follow the next steps:

1. Type `julia` in your terminal and press `]`
2. `] add PotentialLearning.jl`

  Note: this package is not currenlty registered

## How to setup and run your experiment

Load configuration parameters, DFT and reference data, and the potential learning problem.
```julia
    params = get_conf_params("../examples/GaN-SNAP-LAMMPS/")

    # Get DFT data
    dft_train_data, dft_val_data = generate_data("dft", params)

    # Get reference data
    ref_train_data, ref_val_data = generate_data("ref", params)

    # Get potential learning problem (e.g. A β = b)
    snap = learning_problem(dft_train_data, ref_train_data, params)

```

Fit the potentials, forces, and stresses against the DFT and reference data using the configuration parameters.
```julia
    # Solve potential learning problem (e.g. β = A \ b)
    learn(snap, params)
```

Validate trained potentials, forces, and stresses.
```julia
    # Validate potentials, forces, and stresses
    rel_error = validate(snap, dft_val_data - ref_val_data, params)
```


[![Build Status](https://github.com/CESMIX-MIT/PotentialLearning.jl/workflows/CI/badge.svg)](https://github.com/CESMIX-MIT/PotentialLearning.jl/actions)
[![Coverage](https://codecov.io/gh/CESMIX-MIT/PotentialLearning.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/CESMIX-MIT/PotentialLearning.jl)

