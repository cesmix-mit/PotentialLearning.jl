# Load configuration parameters
function load_conf_params(path)
    @show path
    params = Dict()
    open(string(path, "/PotentialLearning.conf")) do f
        while !eof(f)
            line = split(readline(f))
            lhs = line[1]
            if lhs == "rcut"
                rhs = parse(Float64, line[2])
            elseif lhs == "DFT_model"
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
                                                    params["no_atomic_conf"])
    return params
end

# Read atomic positions per configuration
function load_positions_per_conf(path, no_atoms_per_conf, no_atomic_conf)
    @show no_atoms_per_conf, no_atomic_conf
    positions_per_conf = []
    for j = 1:no_atomic_conf
        rs = []
        open(string(path, "/DATA/", string(j), "/DATA")) do f
            for i = 1:23
                readline(f)
            end
            for i = 1:no_atoms_per_conf
                s = split(readline(f))
                r = Point(parse(Float64, s[3]),
                          parse(Float64, s[4]),
                          parse(Float64, s[5]))
                push!(rs, r)
            end
        end
        push!(positions_per_conf, rs)
    end
    return positions_per_conf
end



