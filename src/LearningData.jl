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
    linearize(data::Vector{Force})

Linearize a vector of forces.
E.g. [Force(1.0,2.0,3.0), Force(4.0,5.0,6.0)]  => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
"""
function linearize(data::Vector{Vector{Force}})
    lin_data = Vector{Float64}()
    for i = 1:length(data), j = 1:length(data[i]), k = 1:3
        push!(lin_data, data[i][j][k])
    end
    return lin_data
end


"""
    gen_learning_data(p::Potential, positions_per_conf::Vector,
                      a::Int64, b::Int64, rcut::Float64, fit_forces::Bool)

Generates learning data from a mathematical model. It is used to generate surrogate 
DFT data, or to generate "reference" data (see SNAP mathematical formulation).
"""
function gen_learning_data(p::Potential, positions_per_conf::Vector,
                           a::Int64, b::Int64, rcut::Float64, fit_forces::Bool)
    potentials  = [potential_energy(p, positions_per_conf[j], rcut) for j = a:b]
    
    lin_forces = fit_forces ? linearize([forces(p, positions_per_conf[j], rcut)
                                         for j = a:b]) : Vector{Float64}()
    
    return [potentials; lin_forces]
end


"""
    get_dft_data(params::Dict)

Get DFT data from a surrogate mathematical model, or from an actual DFT
simulation (e.g. using DFTK.jl)
"""
function get_dft_data(params::Dict)
    # Get DFT data from a surrogate mathematical model
    if haskey(params, "DFT_model")
        # Load the potential model (E.g. GaN)
        dft_model = Symbol(params["DFT_model"])
        p = @eval $dft_model($params)
        # Load parameters
        fit_forces = params["fit_forces"]
        a = params["no_train_atomic_conf"]
        b = params["no_atomic_conf"]
        positions_per_conf = params["positions_per_conf"]
        rcut = params["rcut"]
        # Calculate DFT training data
        training_data = gen_learning_data(p, positions_per_conf, 1, a, rcut, fit_forces)
        # Calculate DFT validation data
        validation_data = gen_learning_data(p, positions_per_conf, a+1, b, rcut, fit_forces)
        return training_data, validation_data
    
    else # Get DFT data from an actual DFT simulation (e.g. using DFTK.jl)
        # TODO
        return [], []
    end
end


