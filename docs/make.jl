    pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add PotentialLearning to environment stack

using PotentialLearning
using Documenter
using DocumenterCitations
using Literate

DocMeta.setdocmeta!(PotentialLearning, :DocTestSetup, :(using PotentialLearning); recursive = true)

bib = CitationBibliography(joinpath(@__DIR__, "citations.bib"))

# Generate examples

#const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
#const OUTPUT_DIR = joinpath(@__DIR__, "src/generated")

#examples = Pair{String,String}[]

#for (_, name) in examples
#    example_filepath = joinpath(EXAMPLES_DIR, string(name, ".jl"))
#    Literate.markdown(example_filepath, OUTPUT_DIR, documenter = true)
#end

#examples = [title => joinpath("generated", string(name, ".md")) for (title, name) in examples]

makedocs(bib;
    modules = [PotentialLearning],
    authors = "CESMIX-MIT",
    repo = "https://github.com/cesmix-mit/PotentialLearning.jl/blob/{commit}{path}#{line}",
    sitename = "PotentialLearning.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://cesmix-mit.github.io/PotentialLearning.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
    doctest = true,
    linkcheck = true,
    strict = true
)

deploydocs(;
    repo = "github.com/cesmix-mit/PotentialLearning.jl",
    devbranch = "refactor", # main
    push_preview = true
)
