## PotentialLearning.jl 

Optimize your atomistic data and interatomic potential models in your molecular dynamic workflows.


<ins>**Reduce expensive ***Density functional theory*** calculations**</ins>  while maintaining training accuracy by intelligently subsampling your atomistic dataset:

1) Subsample your [atomistic configurations](https://github.com/JuliaMolSim/AtomsBase.jl) using a Determinantal Point Process ([DPP](https://github.com/dahtah/Determinantal.jl)) based algorithm that compares energy descriptors computed with the Atomic Cluster Expansion ([ACE](https://github.com/ACEsuit)).
```julia
ds = DataSet(conf_train .+ e_descr)
dataset_selector = kDPP(ds, GlobalMean(), DotProduct())
inds = get_random_subset(dataset_selector)
conf_train = @views conf_train[inds]
```
2) Export the reduced dataset, use Density functional theory ([DFT](https://docs.dftk.org/stable/)) on it, and fit your model.

See [example](https://cesmix-mit.github.io/PotentialLearning.jl/dev/generated/DPP-ACE-aHfO2-1/fit-dpp-ace-ahfo2/).

We are working to provide different intelligent subsampling algorithms based on [DPP](https://github.com/dahtah/Determinantal.jl), [DBSCAN](https://docs.google.com/document/d/1SWAanEWQkpsbr2lqetMO3uvdX_QK-Z7dwrgPaM1Dl0o/edit), and [CUR](https://github.com/JuliaLinearAlgebra/LowRankApprox.jl); highly scalable parallel subsampling via hierarchical subsampling and [distributed parallelism](https://github.com/JuliaParallel/Dagger.jl); and optimal subsampler selection.

<ins>**Get fast and accurate interatomic potential models**</ins>  through parallel multi-objective hyper-parameter optimization:

1) Define the interatomic potential model, hyper-parameter value ranges, and custom loss function. Then, [optimize](https://github.com/baggepinnen/Hyperopt.jl) your model.
```julia
model = ACE
pars = OrderedDict( :body_order        => [2, 3, 4],
                    :polynomial_degree => [3, 4, 5], ...)
function custom_loss(metrics::OrderedDict)
    ...
    return w_e * e_mae + w_f * f_mae + w_t * time_us
end
iap, res = hyperlearn!(model, pars, conf_train; loss = custom_loss);
```
2) Export optimal values to your molecular dynamic workflow.

See [example](https://cesmix-mit.github.io/PotentialLearning.jl/dev/generated/Opt-ACE-aHfO2/fit-opt-ace-ahfo2/).

The models are compatible with the interfaces of our sister package [InteratomicPotentials.jl](https://github.com/cesmix-mit/InteratomicPotentials.jl). In particular, we are interested in maintaining compatibility with [ACESuit](https://github.com/ACEsuit), as well as integrating [LAMMPS](https://www.lammps.org/) based potentials such as [ML-POD](https://docs.lammps.org/Packages_details.html#pkg-ml-pod) and [ML-PACE](https://docs.lammps.org/Packages_details.html#ml-pace-package). We are also working to provide neural network potential architecture optimization.

<ins>**Compress your interatomic potential data and model**</ins> using dimensionality reduction of energy and force descriptors:
```julia
pca = PCAState(tol = n_desc)
fit!(ds_train, pca)
transform!(ds_train, pca)
```
See [example](https://cesmix-mit.github.io/PotentialLearning.jl/dev/generated/PCA-ACE-aHfO2/fit-pca-ace-ahfo2/).

We are working to provide feature selection of energy and force descriptors based on [CUR](https://github.com/JuliaLinearAlgebra/LowRankApprox.jl).

Additionally, this package includes utilities for loading input data (such as XYZ files), computing various metrics (including MAE, MSE, RSQ, and COV), exporting results, and generating plots.

**Acknowledgment:** Center for the Exascale Simulation of Materials in Extreme Environments ([CESMIX](https://computing.mit.edu/cesmix/)). Massachusetts Institute of Technology ([MIT](https://www.mit.edu/)).