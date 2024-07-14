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

ENV["BASE_PATH"]    = joinpath(@__DIR__, "../") 

# Citations ####################################################################
bib = CitationBibliography(joinpath(@__DIR__, "citation.bib"))


# Generate examples ############################################################
const examples_path = joinpath(@__DIR__, "..", "examples")
const output_path   = joinpath(@__DIR__, "src/generated")
function create_examples(examples, examples_path, output_path)
    for (_, example_path) in examples
        s = split(example_path, "/")
        sub_path, file_name = string(s[1:end-1]...), s[end]
        example_filepath = joinpath(examples_path, example_path)
        Literate.markdown(example_filepath,
                          joinpath(output_path, sub_path),
                          documenter = true)
    end
    examples = [title => joinpath("generated", replace(example_path, ".jl" => ".md"))
                for (title, example_path) in examples]
    return examples
end

# Basic examples
examples = [
    "Fit a-HfO2 dataset with ACE" => "ACE-aHfO2/fit-ace-ahfo2.jl",
]
basic_examples = create_examples(examples, examples_path, output_path)

# Subsampling examples
examples = [
    "Subsample a-HfO2 dataset with DPP and fit with ACE" => "DPP-ACE-aHfO2-1/fit-dpp-ace-ahfo2.jl",
    "Subsample Na dataset with DPP and fit with ACE" => "DPP-ACE-Na/fit-dpp-ace-na.jl",
    "Subsample Si dataset with DPP, fit with ACE, and cross validate" => "DPP-ACE-Si/fit-dpp-ace-si.jl",
]
ss_examples = create_examples(examples, examples_path, output_path)

# Optimization examples
examples = [
    "Optimize ACE hyper-parameters: minimize force time and fitting error" => "Opt-ACE-aHfO2/fit-opt-ace-ahfo2.jl",
]
opt_examples = create_examples(examples, examples_path, output_path)

# Dimension reduction examples
examples = [
    "Reduce ACE descriptors with PCA and fit a-HfO2 dataset" => "PCA-ACE-aHfO2/fit-pca-ace-ahfo2.jl",
]
dr_examples = create_examples(examples, examples_path, output_path)

examples = [
    "Load Ar+Lennard-Jones dataset and postprocess" => "LJ-Ar/lennard-jones-ar.jl"
]
misc_examples = create_examples(examples, examples_path, output_path)


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
               "Install and run examples" => "install-and-run-examples.md",
               "Basic examples" => basic_examples,
               "Intelligent subsampling" => ss_examples,
               "Hyper-parameter optimization" => opt_examples,
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

