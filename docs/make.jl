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
    "Subsample Na dataset with DPP and fit with ACE" => "DPP-ACE-Na/fit-dpp-ace-na.jl",
    "Load Ar+Lennard-Jones dataset and postprocess" => "LJ-Ar/lennard-jones-ar.jl"
]

for (_, example_path) in examples
    s = split(example_path, "/")
    sub_path, file_name = string(s[1:end-1]...), s[end]
    example_filepath = joinpath(EXAMPLES_DIR, example_path)
    Literate.markdown(example_filepath,
                      joinpath(OUTPUT_DIR, sub_path),
                      documenter = true)
end

examples = [title => joinpath("generated", replace(example_path, ".jl" => ".md"))
            for (title, example_path) in examples]


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
