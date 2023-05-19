# How to run an example

Change the directory to the desired example folder. E.g.
```bash
$ cd PotentialLearning.jl/examples/Na
```

Open Julia REPL, activating ```Project.toml``` file in folder ```examples```, and chosing the number of threads.
```bash
$ julia --project=.. --threads=4
```

Instantiate: type ```]``` in Julia REPL, and then run ```instantiate```.
```julia
    pkg> instantiate
```

Include example script. E.g.
```julia
    julia> include("fit-dpp-ace-na.jl")
```
