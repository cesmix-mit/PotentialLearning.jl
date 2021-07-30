"""
    load_positions_per_conf(path::String, no_atoms_per_conf::Int64,
                            no_conf_init::Int64, no_conf_end::Int64)

Loads atomic positions per configuration.
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
    get_conf_params(path::String)

Loads configuration parameters.
"""
function get_conf_params(path::String)
    params = Dict()
    global key = ""
    open(string(path, "./PotentialLearning.conf")) do f
        while !eof(f)
            line = split(readline(f))
            if length(line) > 0 && line[1][1] != '#' && line[1][1] != ' '
                lhs = line[1]
                if lhs[1] == '['
                    global key = replace(lhs, r"\[(\w+)\]"=>s"\1")
                    global params[key] = Dict()
                else
                    rhs = []
                    for s in line[2:end]
                        if tryparse(Float64, s) !== nothing # if s is a number
                            s = occursin(".", s) ? parse(Float64, s) : parse(Int64, s)
                        elseif tryparse(Bool, s) !== nothing
                            s = parse(Bool, s)
                        end
                        push!(rhs, s)
                    end
                    rhs = length(rhs) == 1 ? rhs[1] : rhs
                    params[key][lhs] = rhs
               end
            end
        end
    end
    params["global"]["path"] = path
    params["global"]["positions_per_conf"] = load_positions_per_conf(path,
                                                params["global"]["no_atoms_per_conf"],
                                                1, params["global"]["no_atomic_conf"])
    return params
end


