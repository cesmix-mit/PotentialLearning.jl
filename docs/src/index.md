# [WIP] PotentialLearning.jl

An open source Julia library for active learning of interatomic potentials in atomistic simulations of materials. It incorporates elements of bayesian inference, machine learning, differentiable programming,  software composability, and high-performance computing. This package is part of a software suite developed for the [CESMIX](https://computing.mit.edu/cesmix/) project.

## Specific goals
- Intelligent data subsampling: iteratively query a large pool of unlabeled data to extract a minimum number of training data that would lead to a supervised ML model with superior accuracy compared to a training model with educated handpicking.
    - Via [DPP](), [clustering](https://docs.google.com/document/d/1SWAanEWQkpsbr2lqetMO3uvdX_QK-Z7dwrgPaM1Dl0o/edit).
- Inference of the optimal values and uncertainties of the model parameters, to propagate them through the atomistic simulation.
    - Interatomic potential hyper-parameter optimization. E.g.  estimation of the optimum cutoff radius.
    - Interatomic potential fitting. The potentials addressed in this package are defined in [InteratomicPotentials.jl](https://github.com/cesmix-mit/InteratomicPotentials.jl) and [InteratomicBasisPotentials.jl](https://github.com/cesmix-mit/InteratomicBasisPotentials.jl). E.g. ACE, SNAP, Neural Network Potentials.
- Measurement of QoI sensitivity to individual parameters. 
- Input data management and post-processing.
  - Process input data so that it is ready for training. E.g. read XYZ file with atomic configurations, linearize energies and forces, split dataset into training and testing, normalize data, transfer data to GPU, define iterators, etc.
  - Post-processing: computation of different metrics (MAE, RSQ, COV, etc), saving results, and plotting.


## Leveraging Julia!
- [Software composability](https://julialang.org/) through [multiple dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY). A series of [composable workflows](https://github.com/cesmix-mit/AtomisticComposableWorkflows) is guiding our design and development. We analyzed three of the most representative workflows: classical molecular dynamics (MD), Ab initio MD, and classical MD with active learning. In addition, it facilitates the training of new  potentials defined by the composition of neural networks with state-of-the-art interatomic potential descriptors.
- [Differentiable programming](https://fluxml.ai/blog/2019/02/07/what-is-differentiable-programming.html). Powerful automatic differentiation tools, such as [Enzyme](https://enzyme.mit.edu/julia/) or [Zygote](https://fluxml.ai/Zygote.jl/latest/), help to accelerate the development of new interatomic potentials by automatically calculating loss function gradients and forces.
- [SciML](https://sciml.ai/): Open Source Software for Scientific Machine Learning. It provides libraries, such as [Optimization.jl](https://github.com/SciML/Optimization.jl), that bring together several optimization packages into one unified Julia interface. 
- Machine learning and HPC abstractions: [Flux.jl](https://fluxml.ai/Flux.jl/stable/) makes parallel learning simple using the NVIDIA GPU abstractions of [CUDA.jl](https://cuda.juliagpu.org/stable/). Mini-batch iterations on heterogeneous data, as required by a loss function based on energies and forces, can be handled by [DataLoader.jl](https://fluxml.ai/Flux.jl/v0.10/data/dataloader/).


## Examples

See [AtomisticComposableWorkflows](https://github.com/cesmix-mit/AtomisticComposableWorkflows) repository. It aims to gather easy-to-use CESMIX-aligned case studies, integrating the latest developments of the Julia atomistic ecosystem with state-of-the-art tools.

