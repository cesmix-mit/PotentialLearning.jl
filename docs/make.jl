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

# Citations ####################################################################

bib = CitationBibliography(joinpath(@__DIR__, "citation.bib"))

# Generate examples ############################################################

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

function create_examples(examples, EXAMPLES_DIR, OUTPUT_DIR)
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
    return examples
end

# Basic examples
examples = [
    "Example 1 - Fit a-HfO2 dataset with ACE" => "ACE-aHfO2/fit-ace-ahfo2.jl",
]
basic_examples = create_examples(examples, EXAMPLES_DIR, OUTPUT_DIR)

# Subsampling examples
examples = [
    "Example 1 - Subsample a-HfO2 dataset with DPP and fit with ACE" => "DPP-ACE-aHfO2-1/fit-dpp-ace-ahfo2.jl",
    "Example 2 - Subsample Na dataset with DPP and fit with ACE" => "DPP-ACE-Na/fit-dpp-ace-na.jl",
    "Example 3 - Subsample Si dataset with DPP, fit with ACE, and cross validate" => "DPP-ACE-Si/fit-dpp-ace-si.jl",
]
ss_examples = create_examples(examples, EXAMPLES_DIR, OUTPUT_DIR)

# Optimization examples
examples = [
    "Example 1 - Optimize ACE hyper-parameters: minimize force time and fitting error" => "Opt-ACE-aHfO2/fit-opt-ace-ahfo2.jl",
]
opt_examples = create_examples(examples, EXAMPLES_DIR, OUTPUT_DIR)

# Dimension reduction examples
examples = [
    "Example 1 - Reduce ACE descriptors with PCA and fit a-HfO2 dataset" => "PCA-ACE-aHfO2/fit-pca-ace-ahfo2.jl",
]
dr_examples = create_examples(examples, EXAMPLES_DIR, OUTPUT_DIR)

examples = [
    "Example 1 - Load Ar+Lennard-Jones dataset and postprocess" => "LJ-Ar/lennard-jones-ar.jl"
]
misc_examples = create_examples(examples, EXAMPLES_DIR, OUTPUT_DIR)


# Make and deploy docs #########################################################

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
               "Basic examples" => basic_examples,
               "Intelligent subsampling" => ss_examples,
               "Hyper-paramter optimization" => opt_examples,
               "Dimension reduction" => dr_examples,
               "Misc" => misc_examples,
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

