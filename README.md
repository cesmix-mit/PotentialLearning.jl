# [WIP] PotentialLearning.jl: The Julia Library of Molecular Dynamics Potentials

In this project, we aim to develop an open-source code for active training, fast calculation, and uncertainty quantification of molecular dynamics potentials for atomistic simulations of materials. 

## Upcoming features
- Surrogate DFT data generation
- Integration with GalacticOptim.jl to perform the optimization process
- Integration with LAMMPS.jl to access the SNAP implementation of LAMMPS
- Implementation of a pure Julia version of SNAP
  - GPU implementation using KernelAbstractions.jl
- Uncertainty quantification of trained potentials
- Open Source

## Installation instructions

To install PotentialLearning.jl in Julia follow the next steps:

1. Type `julia` in your terminal and press `]`
2. `] add PotentialLearning.jl`

## How to setup and run your experiment

Load learning parameters, DFT data, and potential.
```
    # Load learning parameters
    path = "../examples/GaN-SNAP-LAMMPS/"
    learning_params = load_learning_params(path)
    
    # Load DFT data
    dft_training_data, dft_validation_data = load_dft_data(learning_params)
    
    # Load potential
    p_snap = SNAP_LAMMPS(learning_params)
```

Fit the potential against the DFT data using the learning parameters.
```
    # Learn potential, forces, and stresses
    learn(p_snap, dft_training_data, learning_params)

```

Validation of the trained potential, forces, and stresses
```
    # Validate potential, forces, and stresses
    rel_error = validate(p_snap, dft_validation_data, learning_params)
    
```


[![Build Status](https://github.com/CESMIX-MIT/PotentialLearning.jl/workflows/CI/badge.svg)](https://github.com/CESMIX-MIT/PotentialLearning.jl/actions)
[![Coverage](https://codecov.io/gh/CESMIX-MIT/PotentialLearning.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/CESMIX-MIT/PotentialLearning.jl)
