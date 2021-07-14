# [WIP] PotentialLearning.jl: The Julia Library of Molecular Dynamics Potentials

We aim to develop an Open Source code for active training and fast calculation of molecular dynamics potentials for atomistic simulations of materials. 

## Features under development...
- Surrogate DFT data generation
  - Gallium nitride model
- Integration with GalacticOptim.jl to perform the optimization process
- Integration with LAMMPS.jl to access the SNAP implementation of LAMMPS
- Implementation of a pure Julia version of SNAP
  - GPU implementation using KernelAbstractions.jl

## Installation instructions...

To install PotentialLearning.jl in Julia follow the next steps:

1. Type `julia` in your terminal and press `]`
2. `] add PotentialLearning.jl`

  Note: this package is not currenlty registered

## How to setup and run your experiment...

Load configuration parameters, DFT data, and potential.
```julia
    path = "../examples/GaN-SNAP-LAMMPS/"
    params = load_conf_params(path)
    
    dft_training_data, dft_validation_data = load_dft_data(params)
    
    snap = SNAP_LAMMPS(params)
```

Fit the potentials, forces, and stresses against the DFT data using the configuration parameters.
```julia
    learn(snap, dft_training_data, params)

```

Validate trained potentials, forces, and stresses
```julia
    rel_error = validate(snap, dft_validation_data, params)
    
```


[![Build Status](https://github.com/CESMIX-MIT/PotentialLearning.jl/workflows/CI/badge.svg)](https://github.com/CESMIX-MIT/PotentialLearning.jl/actions)
[![Coverage](https://codecov.io/gh/CESMIX-MIT/PotentialLearning.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/CESMIX-MIT/PotentialLearning.jl)
