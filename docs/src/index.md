# [WIP] PotentialLearning.jl

An open source Julia library to enhance active learning of interatomic potentials in atomistic simulations of materials. It incorporates elements of bayesian inference, machine learning, differentiable programming,  software composability, and high-performance computing. This package is part of a software suite developed for the [CESMIX](https://computing.mit.edu/cesmix/) project.

## Specific goals

- **Intelligent subsampling** of atomistic configurations via [DPP](https://github.com/dahtah/Determinantal.jl) and [clustering](https://docs.google.com/document/d/1SWAanEWQkpsbr2lqetMO3uvdX_QK-Z7dwrgPaM1Dl0o/edit)-based algorithms.
- **Optimization of interatomic potentials**
    - Parallel optimization of hyperparameters and coefficients via [Hyperopt.jl](https://github.com/baggepinnen/Hyperopt.jl).
    - Multi-objective optimization (Pareto fronts): force execution time vs fitting accuracy (e.g. MAE of energies and forces).
- **Neuralization of linear interatomic potentials**
    - Neural version of [Julia-ACE](https://github.com/ACEsuit/ACE1.jl) and [LAMMPS-POD](https://docs.lammps.org/pair_pod.html).
    - Integration with [Flux](https://fluxml.ai/Flux.jl/stable/) ecosystem.
- **Interatomic potential compression**
    - Feature selection (e.g. [CUR](https://github.com/JuliaLinearAlgebra/LowRankApprox.jl)) and dimensionality reduction (e.g [PCA](https://juliastats.org/MultivariateStats.jl/dev/pca/)) of atomistic descriptors.
- **Interatomic potential fitting/training**
    - Inference of parameter uncertainties in linear interatomic potentials.
- **Quantity of Interest (QoI) sensitivity** analysis of interatomic potential parameters.
- **Dimension reduction of QoI** through the theory of Active Subspaces.
- **Atomistic configuration and DFT data management and post-processing**
  - Process input data so that it is ready for training. E.g. read XYZ file with atomic configurations, linearize energies and forces, split dataset into training and testing, normalize data, transfer data to GPU, define iterators, etc.
  - Post-processing: computation of different metrics (MAE, RSQ, COV, etc), saving results, and plotting.

## Leveraging Julia!

- [Software composability](https://julialang.org/) through [multiple dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY). A series of [composable workflows](https://github.com/cesmix-mit/AtomisticComposableWorkflows) is guiding our design and development. We analyzed three of the most representative workflows: classical molecular dynamics (MD), Ab initio MD, and classical MD with active learning. In addition, it facilitates the training of new  potentials defined by the composition of neural networks with state-of-the-art interatomic potential descriptors.
- [Differentiable programming](https://fluxml.ai/blog/2019/02/07/what-is-differentiable-programming.html). Powerful automatic differentiation tools, such as [Enzyme](https://enzyme.mit.edu/julia/) or [Zygote](https://fluxml.ai/Zygote.jl/latest/), help to accelerate the development of new interatomic potentials by automatically calculating loss function gradients and forces.
- [SciML](https://sciml.ai/): Open Source Software for Scientific Machine Learning. It provides libraries, such as [Optimization.jl](https://github.com/SciML/Optimization.jl), that bring together several optimization packages into one unified Julia interface. 
- Machine learning and HPC abstractions: [Flux.jl](https://fluxml.ai/Flux.jl/stable/) makes parallel learning simple using the NVIDIA GPU abstractions of [CUDA.jl](https://cuda.juliagpu.org/stable/). Mini-batch iterations on heterogeneous data, as required by a loss function based on energies and forces, can be handled by [DataLoader.jl](https://fluxml.ai/Flux.jl/v0.10/data/dataloader/).

