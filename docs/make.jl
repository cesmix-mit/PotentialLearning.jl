pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add PotentialLearning to environment stack

using PotentialLearning
using Documenter
using DocumenterCitations
using Literate

DocMeta.setdocmeta!(
    PotentialLearning,
    :DocTestSetup,
    :(using PotentialLearning);
    recursive = true,
)


bib = CitationBibliography(joinpath(@__DIR__, "citation.bib"))

# Generate examples

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
    "Subsampling, compute descriptors, and fit ACE" => "Na/fit-dpp-ace-na"
]

for (_, name) in examples
    example_filepath = joinpath(EXAMPLES_DIR, string(name, ".jl"))
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
end

examples = [title=>joinpath("generated", string(name, ".md")) for (title, name) in examples]


makedocs(
      root    =  joinpath(dirname(pathof(PotentialLearning)), "..", "docs"),
      source  = "src",
      build   = "build",
      clean   = true,
      doctest = true,
      modules = [PotentialLearning],
      repo    = "https://github.com/cesmix-mit/PotentialLearning.jl/blob/{commit}{path}#{line}",
      highlightsig = true,
      sitename = "PotentialLearning.jl",
      expandfirst = [],
      draft = false,    
      pages = ["Home" => "index.md",
               "How to run the examples" => "how-to-run-the-examples.md",
               "Examples" => examples,
               "API" => "api.md"],
      format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://cesmix-mit.github.io/PotentialLearning.jl",
        assets = String[],
      ),
      plugins=[bib]
)

deploydocs(;
    repo = "github.com/cesmix-mit/PotentialLearning.jl",
    devbranch = "main",
    push_preview = true,
)
