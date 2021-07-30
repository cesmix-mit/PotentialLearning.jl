
"""
    generate_data(params::Dict)

Generates DFT/reference data based on a particular potential.
"""
function generate_data(model::String, params::Dict)
    # Load the potential model (E.g. GaN)
    p_model = Symbol(params[model]["potential"])
    p = @eval $p_model($(params[model]))
    # Load parameters
    fit_forces = params["global"]["fit_forces"]
    a = params["global"]["no_train_atomic_conf"]
    b = params["global"]["no_atomic_conf"]
    positions_per_conf = params["global"]["positions_per_conf"]
    rcut = params["global"]["rcut"]
    # Calculate DFT training data
    training_data = generate_data_aux(p, positions_per_conf, 1, a, rcut, fit_forces)
    # Calculate DFT validation data
    validation_data = generate_data_aux(p, positions_per_conf, a+1, b, rcut, fit_forces)
    
    return training_data, validation_data
end

"""
    generate_data_aux(p::Potential, positions_per_conf::Vector,
                      a::Int64, b::Int64, rcut::Float64, fit_forces::Bool)

Auxiliar function. See `generate_data`.
"""
function generate_data_aux(p::Potential, positions_per_conf::Vector,
                           a::Int64, b::Int64, rcut::Float64, fit_forces::Bool)
    potentials  = [potential_energy(p, positions_per_conf[j], rcut) for j = a:b]
    lin_forces = fit_forces ? linearize([forces(p, positions_per_conf[j], rcut)
                                         for j = a:b]) : Vector{Float64}()
    return [potentials; lin_forces]
end

"""
    linearize(data::Vector{Force})

Linearizes a vector of forces.
E.g. [Force(1.0,2.0,3.0), Force(4.0,5.0,6.0)] => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
"""
function linearize(data::Vector{Vector{Force}})
    lin_data = Vector{Float64}()
    for i = 1:length(data), j = 1:length(data[i]), k = 1:3
        push!(lin_data, data[i][j][k])
    end
    return lin_data
end

