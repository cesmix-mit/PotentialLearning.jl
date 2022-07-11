
export get_defaults_args, get_input, load_dataset, linearize_forces, get_batches

"""
    get_defaults_args()
    
Returns default input arguments for NN potential training.

TODO: probably this should not be here.

"""
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


"""
    get_input(args)
    
`args`: vector of arguments (strings)

Returns an OrderedDict with the arguments.
See https://github.com/cesmix-mit/AtomisticComposableWorkflows documentation
for information about how to define the input arguments.

"""
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


"""
    load_dataset(n_train_sys, n_test_sys, dataset_path, dataset_filename)
    
`n_train_sys`: no. of training systems
`n_test_sys`: no. of test systems
`dataset_path`: dataset path
`dataset_filename`: dataset filename

Returns training and test energies, forces, and stresses.
The input dataset is split into training and test datasets.

"""
function load_dataset(n_train_sys, n_test_sys, dataset_path, dataset_filename)

    n_sys = n_train_sys + n_test_sys
    filename = dataset_path*dataset_filename
    systems, energies, forces, stress =
                               load_extxyz(filename, max_entries = n_sys)
    rand_list = randperm(n_sys)
    train_index, test_index = rand_list[1:n_train_sys], rand_list[n_train_sys+1:n_sys]
    train_systems, train_energies, train_forces, train_stress =
                                 systems[train_index], energies[train_index],
                                 forces[train_index], stress[train_index]
    test_systems, test_energies, test_forces, test_stress =
                                 systems[test_index], energies[test_index],
                                 forces[test_index], stress[test_index]
    return train_systems, train_energies, train_forces, train_stress,
           test_systems, test_energies, test_forces, test_stress
end


"""
    load_dataset(n_train_sys, n_test_sys, dataset_path,
                 trainingset_filename, testset_filename)
    
`n_train_sys`: no. of training systems
`n_test_sys`: no. of test systems
`dataset_path`: datasets path
`trainingset_filename`: training dataset filename
`testset_filename`: test dataset filename

Returns training and test energies, forces, and stresses.
Training and test datasets are already defined.

"""
function load_dataset(n_train_sys, n_test_sys, dataset_path,
                      trainingset_filename, testset_filename)
    n_sys = n_train_sys + n_test_sys
    filename = dataset_path*trainingset_filename
    train_systems, train_energies, train_forces, train_stress =
            load_extxyz(filename, max_entries = n_train_sys)
    filename = dataset_path*testset_filename
    test_systems, test_energies, test_forces, test_stress =
            load_extxyz(filename, max_entries = n_test_sys)
    return train_systems, train_energies, train_forces, train_stress,
           test_systems, test_energies, test_forces, test_stress
end


"""
    load_dataset(input)
    
`input`: OrderedDict with input arguments. See `get_defaults_args()`.

Returns training and test energies, forces, and stresses.

"""
function load_dataset(input)
    n_train_sys = input["n_train_sys"]
    n_test_sys = input["n_test_sys"]
    dataset_path = input["dataset_path"]
    if "dataset_filename" in keys(input)
        return load_dataset(n_train_sys, n_test_sys, dataset_path,
                            dataset_filename)
    else
        trainingset_filename = input["trainingset_filename"]
        testset_filename = input["testset_filename"]
        return load_dataset(n_train_sys, n_test_sys, dataset_path,
                            trainingset_filename, testset_filename)
    end
end


"""
    linearize_forces(forces)
    
`forces`: vector of forces per system

Returns a vector with the components of the forces of the systems.

"""
function linearize_forces(forces)
    return vcat([vcat(vcat(f...)...) for f in forces]...)
end


"""
    get_batches(n_batches, B_train, B_train_ext, e_train, dB_train, f_train,
                B_test, B_test_ext, e_test, dB_test, f_test)

`n_batches`: no. of batches per dataset.
`B_train`: descriptors of the energies used in training.
`B_train_ext`: extendended descriptors of the energies used in training. Requiered to compute forces.
`e_train`: energies used in training.
`dB_train`: derivatives of the energy descritors used in training.
`f_train`: forces used in training.
`B_test`: descriptors of the energies used in test.
`B_test_ext`: extendended descriptors of the energies used in test. Requiered to compute forces.
`e_test`: energies used in test.
`dB_test`: derivatives of the energy descritors used in test.
`f_test`: forces used in test.

Returns the data loaders for training and test of energies and forces.

"""
function get_batches(n_batches, B_train, B_train_ext, e_train, dB_train, f_train,
                     B_test, B_test_ext, e_test, dB_test, f_test)
    
    bs_train_e = floor(Int, length(B_train) / n_batches)
    train_loader_e   = DataLoader((B_train, e_train), batchsize=bs_train_e, shuffle=true)
    bs_train_f = floor(Int, length(dB_train) / n_batches)
    train_loader_f   = DataLoader((B_train_ext, dB_train, f_train),
                                   batchsize=bs_train_f, shuffle=true)

    bs_test_e = floor(Int, length(B_test) / n_batches)
    test_loader_e   = DataLoader((B_test, e_test), batchsize=bs_test_e, shuffle=true)
    bs_test_f = floor(Int, length(dB_test) / n_batches)
    test_loader_f   = DataLoader((B_test_ext, dB_test, f_test),
                                  batchsize=bs_test_f, shuffle=true)
    
    return train_loader_e, train_loader_f, test_loader_e, test_loader_f
end


