# How to run an example

Change the directory to the desired example folder. E.g.
```bash
$ cd PotentialLearning.jl/examples/Na
```

Open Julia REPL, activate ```Project.toml``` file in folder ```examples```, and chose the number of threads. E.g.
```bash
$ julia --project=.. --threads=4
```

Type ```]``` in Julia REPL, and then run ```instantiate```.
```julia
    pkg> instantiate
```

Include example script. E.g.
```julia
    julia> include("fit-dpp-ace-na.jl")
```
