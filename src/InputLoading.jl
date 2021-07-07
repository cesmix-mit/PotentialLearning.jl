# Load configuration parameters
function load_learning_params(path::String)
    @show path
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

# Load atomic positions per configuration
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

# Get DFT data 
# ToDo: should also load actual DFT data instead of loading only surrogate data.
function load_dft_data(params::Dict)

    path = params["path"]
    rows = params["rows"]
    no_atomic_conf = params["no_atomic_conf"]
    positions_per_conf = params["positions_per_conf"]
    rcut = params["rcut"]
    
    # Load a potential model (E.g. GaN model)
    dft_potential = load_potential(path, params["DFT_model"])
    
    # Calculate potential energy per configuration (vector b) using the potential model
    dft_training_data =  [potential_energy(positions_per_conf[j], rcut, dft_potential) 
                          for j = 1:rows]
                          
    dft_validation_data =  [potential_energy(positions_per_conf[j], rcut, dft_potential) 
                            for j = rows+1:no_atomic_conf]
    
    return dft_training_data, dft_validation_data
end


