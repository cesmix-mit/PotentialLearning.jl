
## PotentialLearning.jl 

Optimize your atomistic data and interatomic potential models in your molecular dynamics workflows.

<!--<a href="https://cesmix-mit.github.io/PotentialLearning.jl/stable">
<img alt="Stable documentation" src="https://img.shields.io/badge/documentation-stable%20release-blue?style=flat-square">
</a>-->
<a href="https://cesmix-mit.github.io/PotentialLearning.jl/dev">
<img alt="Development documentation" src="https://img.shields.io/badge/documentation-in%20development-orange?style=flat-square">
</a>
<a href="https://mit-license.org">
<img alt="MIT license" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square">
</a>
<a href="https://github.com/cesmix-mit/PotentialLearning.jl/issues/new">
<img alt="Ask us anything" src="https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg?style=flat-square">
</a>
</a> 
<br />
<br />

**Reduce expensive Density Functional Theory (DFT) calculations** while maintaining training accuracy by intelligently reducing your atomistic dataset.

```julia
# Reduce your atomistic dataset by intellegently comparing energy descriptors.
ds = DataSet(conf_train .+ e_descr)
dataset_selector = kDPP(ds, GlobalMean(), DotProduct())
inds = get_random_subset(dataset_selector)
conf_train = @views conf_train[inds]
```

***Get fast and accurate interatomic potential models*** through multi-objective hyper-parameter optimization.

```julia
# Define the interatomic potential model and hyper-parameter value ranges.
model = ACE
pars = OrderedDict( :body_order        => [2, 3, 4],
                    :polynomial_degree => [3, 4, 5], ...)

# Define your custom loss function: fitting accuracy and force calculations.
function custom_loss(metrics::OrderedDict)
    ...
    return w_e * e_mae + w_f * f_mae + w_t * time_us
end

# Optimize the hyper-parameters.
iap, res = hyperlearn!(model, pars, conf_train; loss = custom_loss);
```


**Acknowledgment:** Center for the Exascale Simulation of Materials in Extreme Environments [CESMIX](https://computing.mit.edu/cesmix/). Massachusetts Institute of Technology (MIT).


