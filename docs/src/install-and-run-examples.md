# Install package and run examples

## Add registries and package
Open a Julia REPL (`$ julia`), type `]` to enter the Pkg REPL, and add the following registries:
```julia
    pkg> registry add https://github.com/JuliaRegistries/General
    pkg> registry add https://github.com/cesmix-mit/CESMIX.git 
    pkg> registry add https://github.com/JuliaMolSim/MolSim.git
    pkg> registry add https://github.com/ACEsuit/ACEregistry
```

Then, add PotentialLearning:
```julia
    pkg> add PotentialLearning

```

## Clone repository and access an example folder
Clone `PotentialLearning.jl` repository in your working directory.
```shell
    $ git clone git@github.com:cesmix-mit/PotentialLearning.jl.git
```
Access to any folder within `PotentialLearning.jl/examples`. E.g.
```shell
    $ cd PotentialLearning.jl/examples/DPP-ACE-aHfO2-1
```

## Run example
Open a Julia REPL, activate the `examples` folder project, and define the number of threads.
```julia
    $ julia --project=./ --threads=4
```
Type `]` to enter the Pkg REPL and instantiate.
```julia
    pkg> instantiate
```
Finally, include the example file.
```julia
    julia> include("fit-dpp-ace-ahfo2.jl")
```

