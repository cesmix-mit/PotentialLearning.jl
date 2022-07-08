using AtomsBase
using InteratomicPotentials 
using InteratomicBasisPotentials
using OrderedCollections
using IterTools
using LinearAlgebra 
using UnitfulAtomic
using Unitful 
using Flux
using Flux.Data: DataLoader
using BSON: @save

include("load-data.jl")

# Load input parameters

function get_defaults_args()
    args = ["experiment_path",      "TiO2/",
            "dataset_path",         "data/",
            "trainingset_filename", "TiO2trainingset.xyz",
            "testset_filename",     "TiO2testset.xyz",
            "n_train_sys",          "80",
            "n_test_sys",           "20",
            "n_batches",            "8",
            "n_body",               "3",
            "max_deg",              "3",
            "r0",                   "1.0",
            "rcutoff",              "5.0",
            "wL",                   "1.0",
            "csp",                  "1.0",
            "w_e",                  "1.0",
            "w_f",                  "1.0"]
     return args
end

function get_input(args)
    if length(args) == 0
        args = get_defaults_args()
    end
    input = OrderedDict()
    for (key, val) in partition(args,2,2)
        if tryparse(Float64, val) != nothing
            if occursin(".", val)
                val = parse(Float64, val)
            else
                val = parse(Int64, val)
            end
        end
        input[key] = val
    end
    return input
end


# Load dataset
function load_dataset(input)
    experiment_path = input["experiment_path"]
    n_train_sys, n_test_sys = input["n_train_sys"], input["n_test_sys"]
    n_sys = n_train_sys + n_test_sys
    # Split into training and testing
    if "dataset_filename" in keys(input)
        filename = input["dataset_path"]*input["dataset_filename"]
        systems, energies, forces, stress =
                                   load_data(filename, max_entries = n_sys)
        rand_list = randperm(n_sys)
        train_index, test_index = rand_list[1:n_train_sys], rand_list[n_train_sys+1:n_sys]
        train_systems, train_energies, train_forces, train_stress =
                                     systems[train_index], energies[train_index],
                                     forces[train_index], stress[train_index]
        test_systems, test_energies, test_forces, test_stress =
                                     systems[test_index], energies[test_index],
                                     forces[test_index], stress[test_index]
    else # The data set is already split.
        filename = input["dataset_path"]*input["trainingset_filename"]
        train_systems, train_energies, train_forces, train_stress =
                load_data(filename, max_entries = n_train_sys)
        filename = input["dataset_path"]*input["testset_filename"]
        test_systems, test_energies, test_forces, test_stress =
                load_data(filename, max_entries = n_test_sys)
    end
    return train_systems, train_energies, train_forces, train_stress,
           test_systems, test_energies, test_forces, test_stress
end


# Linearize forces
function linearize_forces(forces)
    return vcat([vcat(vcat(f...)...) for f in forces]...)
end


# Split data into batches
function get_batches(n_batches, B_train, B_train_ext, e_train, dB_train, f_train,
                     B_test, B_test_ext, e_test, dB_test, f_test)
    
    bs_train_e = floor(Int, length(B_train) / n_batches)
    train_loader_e   = DataLoader((B_train, e_train), batchsize=bs_train_e, shuffle=true)
    bs_train_f = floor(Int, length(dB_train) / n_batches)
    train_loader_f   = DataLoader((B_train_ext, dB_train, f_train),
                                   batchsize=bs_train_f, shuffle=true)
    println("Batch sizes of training energy and force datasets: ", bs_train_e, ", ", bs_train_f)

    bs_test_e = floor(Int, length(B_test) / n_batches)
    test_loader_e   = DataLoader((B_test, e_test), batchsize=bs_test_e, shuffle=true)
    bs_test_f = floor(Int, length(dB_test) / n_batches)
    test_loader_f   = DataLoader((B_test_ext, dB_test, f_test),
                                  batchsize=bs_test_f, shuffle=true)
    println("Batch sizes of test energy and force datasets: ", bs_train_e, ", ", bs_train_f)
    
    return train_loader_e, train_loader_f, test_loader_e, test_loader_f
end


