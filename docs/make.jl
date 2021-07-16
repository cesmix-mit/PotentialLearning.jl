# Inside make.jl
push!(LOAD_PATH,"../src/")
using PotentialLearning
using Documenter

DocMeta.setdocmeta!(PotentialLearning, :DocTestSetup, :(using PotentialLearning); recursive=true)

makedocs(
        modules  = [PotentialLearning],
        authors = "CESMIX-MIT",
        repo="https://github.com/cesmix-mit/PotentialLearning.jl/blob/{commit}{path}#{line}",
        sitename="PotentialLearning.jl",
        format=Documenter.HTML(;
            prettyurls=get(ENV, "CI", "false") == "true",
            canonical="https://cesmix-mit.github.io/PotentialLearning.jl",
            assets=String[],
        ),
        pages = [
            "Home" => "index.md"
            "Functions" => "functions.md"
        ],
        doctest = true,
        linkcheck = true,
        strict = true,
)

deploydocs(;
    repo="github.com/cesmix-mit/PotentialLearning.jl",
    devbranch = "main",
)
