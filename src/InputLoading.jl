
"""
    load_conf_params(path::String)

Load configuration parameters
"""
function get_conf_params(path::String)
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


