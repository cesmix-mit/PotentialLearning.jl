#export get_input, load_datasets, linearize_forces, get_batches


"""
    to_num(str)
    
`str`: string with a number: integer or float

Returns an integer or float.

"""
function to_num(str)
    val = nothing
    if occursin(".", str)
        val = parse(Float64, str)
    else
        val = parse(Int64, str)
    end
    return val
end


"""
    get_input(args)
    
`args`: vector of arguments (strings)

Returns an OrderedDict with the arguments.
See https://github.com/cesmix-mit/AtomisticComposableWorkflows documentation
for information about how to define the input arguments.

"""
function get_input(args)
    input = OrderedDict()
    for (key, val) in partition(args,2,2)
        val = replace(val, " " => "")
        # if val is a boolean
        if val == "true" || val == "false"
            val = val == "true"
        # if val is a vector, e.g. "[1.5,1.5]"
        elseif val[1] == '['
            val = to_num.(split(val[2:end-1], ","))
        # if val is a number, e.g. 1.5 or 1
        elseif tryparse(Float64, val) != nothing
            val = to_num(val)
        end
        input[key] = val
    end
    return input
end


"""
    load_datasets(input)
    
`input`: OrderedDict with input arguments. See `get_defaults_args()`.

Returns training and test systems, energies, forces, and stresses.

"""
function load_datasets(input)
    if "dataset_filename" in keys(input) # Load and split dataset
        # Load dataset
        filename = input["dataset_path"]*input["dataset_filename"]
        systems, energies, forces, stress = load_extxyz(filename)
        # Split dataset
        split_prop = input["split_prop"]
        n_sys = length(systems)
        n_train_sys = round(Int, split_prop * n_sys)
        n_test_sys = n_sys - n_train_sys
        rand_list = randperm(n_sys)
        train_index, test_index = rand_list[1:n_train_sys], rand_list[n_train_sys+1:n_sys]
        train_sys, train_e, train_f, train_s =
                                     systems[train_index], energies[train_index],
                                     forces[train_index], stress[train_index]
        test_sys, test_e, test_f, test_s =
                                     systems[test_index], energies[test_index],
                                     forces[test_index], stress[test_index]
    else # Load training and test datasets
        filename = input["dataset_path"]*input["trainingset_filename"]
        train_sys, train_e, train_f, train_s = load_extxyz(filename)
        filename = input["dataset_path"]*input["testset_filename"]
        test_sys, test_e, test_f, test_s = load_extxyz(filename)
    end
    
    return train_sys, train_e, train_f, train_s,
           test_sys, test_e, test_f, test_s
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


