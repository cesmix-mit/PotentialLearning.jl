# How to run the examples

## Add registries

Open a Julia REPL (`$ julia`), type `]` to enter the Pkg REPL, and add the following registries:
```julia
    pkg> registry add https://github.com/JuliaRegistries/General
    pkg> registry add https://github.com/cesmix-mit/CESMIX.git 
    pkg> registry add https://github.com/JuliaMolSim/MolSim.git
    pkg> registry add https://github.com/ACEsuit/ACEregistry
```

## Install the dependencies of the `examples` folder project

After cloning the `PotentialLearning.jl` repository in your working directory (`$ git clone git@github.com:cesmix-mit/PotentialLearning.jl.git`), and activate the `examples`  folder project.
```shell
    $ julia --project=PotentialLearning.jl/examples
```
 Type `]` to enter the Pkg REPL and instantiate.
```julia
    pkg> instantiate
```

## Run an example

Access to any folder within `PotentialLearning.jl/examples`. E.g.
```shell
    $ cd PotentialLearning.jl/examples/ACE
```
Open a Julia REPL, activate the `examples` folder project, and define the number of threads.
```julia
    $ julia --project=../ --threads=4
```
Finally, include the example file.
```julia
    julia> include("fit-ace.jl")
```
