- InteartomicBasisPotentialExtensions.jl
    Functions to compute descriptors in parallel using threads

- PotentialLearningExtensions.jl
    - Function to split dataset
    - Function to fit ACE using normal equations
    - Function to get all energies and forces based on a dataset
    - Modification to the function LinearProblem (see linear.jl)
        - true, sum.(get_values.(get_local_descriptors.(ds))), get_values.(get_energy.(ds))
        
- macros.jl
    - Macros to facilitate saving files

- metrics.jl
    - Functions to compute different performance metrics (mae, rmse, rsqr, etc)
    
- plots
    - Functions to plot energies, forces, and cosine of forces.
    
- ace.jl
    - This file improve performance of descriptor calculation in InteratomicBasisPotentials.jl. I should open a PR.

