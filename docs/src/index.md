# [WIP] PotentialLearning.jl

**Developing optimization workflows for fast and accurate interatomic potentials**. This package is part of a software suite developed for the [CESMIX](https://computing.mit.edu/cesmix/) project.

## Goals

**Optimize your atomistic data:** intelligent subsampling of large datasets to reduce DFT computations
- Intelligent subsampling of atomistic configurations using algorithms based on [DPP](https://github.com/dahtah/Determinantal.jl), [DBSCAN](https://docs.google.com/document/d/1SWAanEWQkpsbr2lqetMO3uvdX_QK-Z7dwrgPaM1Dl0o/edit), [CUR](https://github.com/JuliaLinearAlgebra/LowRankApprox.jl), etc.
- Highly scalable parallel subsampling via hierarchical subsampling and distributed parallelism ([Dagger.jl](https://github.com/JuliaParallel/Dagger.jl)).
- Optimal subsampler choosing via [Hyperopt.jl](https://github.com/baggepinnen/Hyperopt.jl).

**Optimize your interatomic potential model:** hyperparameters, coefficients, model compression, and model selection.
- Parallel optimization of hyperparameters, coefficients, and model selection via [Hyperopt.jl](https://github.com/baggepinnen/Hyperopt.jl); multi-objective optimization (Pareto fronts): force execution time vs fitting accuracy (e.g. MAE of energies and forces).
- Model compression via feature selection (e.g. [CUR](https://github.com/JuliaLinearAlgebra/LowRankApprox.jl)) and dimensionality reduction (e.g [PCA](https://juliastats.org/MultivariateStats.jl/dev/pca/), Active Subspaces) of atomistic descriptors.
- Fitting of linear potentials and inference of parameter uncertainties. Training of neural versions of [Julia-ACE](https://github.com/ACEsuit/ACE1.jl) and [LAMMPS-POD](https://docs.lammps.org/pair_pod.html).

Additionally, this package provides utilities for atomistic configuration and DFT data management and post-processing.
  - Process input data so that it is ready for training. E.g. read XYZ file with atomic configurations, linearize energies and forces, split dataset into training and testing, normalize data, transfer data to GPU, define iterators, etc.
  - Post-processing: computation of different metrics (MAE, RSQ, COV, etc), saving results, and plotting.

## Leveraging Julia!

- [Software composability](https://julialang.org/) through [multiple dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY). A series of [composable workflows](https://github.com/cesmix-mit/AtomisticComposableWorkflows) is guiding our design and development. We analyzed three of the most representative workflows: classical molecular dynamics (MD), Ab initio MD, and classical MD with active learning. In addition, it facilitates the training of new  potentials defined by the composition of neural networks with state-of-the-art interatomic potential descriptors.
- [Differentiable programming](https://fluxml.ai/blog/2019/02/07/what-is-differentiable-programming.html). Powerful automatic differentiation tools, such as [Enzyme](https://enzyme.mit.edu/julia/) or [Zygote](https://fluxml.ai/Zygote.jl/latest/), help to accelerate the development of new interatomic potentials by automatically calculating loss function gradients and forces.
- [SciML](https://sciml.ai/): Open Source Software for Scientific Machine Learning. It provides libraries, such as [Optimization.jl](https://github.com/SciML/Optimization.jl), that bring together several optimization packages into one unified Julia interface. 
- Machine learning and HPC abstractions: [Flux.jl](https://fluxml.ai/Flux.jl/stable/) makes parallel learning simple using the NVIDIA GPU abstractions of [CUDA.jl](https://cuda.juliagpu.org/stable/). Mini-batch iterations on heterogeneous data, as required by a loss function based on energies and forces, can be handled by [DataLoader.jl](https://fluxml.ai/Flux.jl/v0.10/data/dataloader/).

