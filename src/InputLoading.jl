
"""
    load_conf_params(path::String)

Load configuration parameters
"""
function load_conf_params(path::String)
    params = Dict()
    params["path"] = path
    open(string(path, "/PotentialLearning.conf")) do f
        while !eof(f)
            line = split(readline(f))
            lhs = line[1]
            if lhs == "rcut"
                rhs = parse(Float64, line[2])
            elseif lhs == "DFT_model" || lhs == "solver"
                rhs = line[2]
            elseif length(line[2:end]) == 1
                rhs = parse(Int64, line[2])
            else
                rhs = parse.(Int64, line[2:end])
            end
            params[lhs] = rhs
        end
    end 
    params["positions_per_conf"] = load_positions_per_conf(path,
                                                    params["no_atoms_per_conf"],
                                                    1, params["no_atomic_conf"])
    return params
end

"""
    load_positions_per_conf(path::String, no_atoms_per_conf::Int64,
                                 no_conf_init::Int64, no_conf_end::Int64)

Load atomic positions per configuration
"""
function load_positions_per_conf(path::String, no_atoms_per_conf::Int64,
                                 no_conf_init::Int64, no_conf_end::Int64)
    positions_per_conf = []
    for j = no_conf_init:no_conf_end
        rs = Vector{Position}()
        open(string(path, "/DATA/", string(j), "/DATA")) do f
            for i = 1:23
                readline(f)
            end
            for i = 1:no_atoms_per_conf
                s = split(readline(f))
                r = Position(parse(Float64, s[3]),
                             parse(Float64, s[4]),
                             parse(Float64, s[5]))
                push!(rs, r)
            end
        end
        push!(positions_per_conf, rs)
    end
    return positions_per_conf
end


"""
    load_dft_data(params::Dict)

Load DFT data 
"""
function load_dft_data(params::Dict)
    # ToDo: should also load actual DFT data instead of loading only surrogate data.
    path = params["path"]
    no_train_atomic_conf = params["no_train_atomic_conf"]
    no_atomic_conf = params["no_atomic_conf"]
    positions_per_conf = params["positions_per_conf"]
    rcut = params["rcut"]
    
    # Load a potential model (E.g. GaN)
    dft_model = Symbol(params["DFT_model"])
    dft_potential = @eval $dft_model($params)
    
    # Calculate DFT training data
    potential_dft_data = [potential_energy(dft_potential, positions_per_conf[j], rcut)
                          for j = 1:no_train_atomic_conf]
    force_dft_data = [forces(dft_potential, positions_per_conf[j], rcut)
                      for j = 1:no_train_atomic_conf]
    force_dft_data_lin = Vector{Float64}()
    for i = 1:length(force_dft_data), j = 1:length(force_dft_data[i]), k = 1:3
        push!(force_dft_data_lin, force_dft_data[i][j][k])
    end
    dft_training_data = [potential_dft_data; force_dft_data_lin]

    # Calculate DFT validation data
    potential_dft_data = [potential_energy(dft_potential, positions_per_conf[j], rcut)
                          for j = no_train_atomic_conf+1:no_atomic_conf]
    force_dft_data = [forces(dft_potential, positions_per_conf[j], rcut)
                      for j = no_train_atomic_conf+1:no_atomic_conf]
    force_dft_data_lin = Vector{Float64}()
    for i = 1:length(force_dft_data), j = 1:length(force_dft_data[i]), k = 1:3
        push!(force_dft_data_lin, force_dft_data[i][j][k])
    end
    dft_validation_data = [potential_dft_data; force_dft_data_lin]

    return dft_training_data, dft_validation_data
end


