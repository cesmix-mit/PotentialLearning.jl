import PotentialLearning

# Get input parameters from OrderedDict

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



# Split datasets

function Base.split(ds, n, m)
    ii = randperm(length(ds))
    return @views ds[first(ii, n)], ds[last(ii, m)]
end


# Learning function using normal equations

function learn!(lp, w_e, w_f)

    @views B_train = reduce(hcat, lp.B)'
    @views dB_train = reduce(hcat, lp.dB)'
    @views e_train = lp.e
    @views f_train = reduce(vcat, lp.f)
    
    # Calculate A and b.
    @views A = [B_train; dB_train]
    @views b = [e_train; f_train]

    # Calculate coefficients β.
    Q = Diagonal([w_e * ones(length(e_train));
                  w_f * ones(length(f_train))])
    β = (A'*Q*A) \ (A'*Q*b)

    copyto!(lp.β, β)
end


# Auxiliary functions to compute all energies and forces as vectors (Zygote-friendly functions)

function get_all_energies(ds::DataSet)
    return [get_values(get_energy(ds[c])) for c in 1:length(ds)]
end

function get_all_forces(ds::DataSet)
    return reduce(vcat,reduce(vcat,[get_values(get_forces(ds[c]))
                                    for c in 1:length(ds)]))
end

function get_all_energies(ds::DataSet, lp::PotentialLearning.LinearProblem)
    Bs = sum.(get_values.(get_local_descriptors.(ds)))
    return dot.(Bs, [lp.β])
end

function get_all_forces(ds::DataSet, lp::PotentialLearning.LinearProblem)
    force_descriptors = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds]
    return vcat([dB' * lp.β for dB in [reduce(hcat, fi) for fi in force_descriptors]]...)
end

#function get_all_energies(ds::DataSet, nnbp::NeuralNetworkBasisPotential)
#    return [potential_energy(ds[c], nnbp) for c in 1:length(ds)]
#end

#function get_all_forces(ds::DataSet, nnbp::NeuralNetworkBasisPotential)
#    return reduce(vcat,reduce(vcat,[force(ds[c], nnbp)
#                                    for c in 1:length(ds)]))
#end

# Definition of a linear problem. Changes descriptors calculations. TODO: open PR?
"""
    LinearProblem(ds::DatasSet; T = Float64)

Construct a LinearProblem by detecting if there are energy descriptors and/or force descriptors and construct the appropriate LinearProblem (either Univariate, if only a single type of descriptor, or Covariate, if there are both types).
"""
function PotentialLearning.LinearProblem(ds::DataSet; T = Float64)
    d_flag, descriptors, energies = try
        #true,  compute_features(ds, GlobalSum()), get_values.(get_energy.(ds))
        true, sum.(get_values.(get_local_descriptors.(ds))), get_values.(get_energy.(ds))
        
        #true, compute_feature.(get_local_descriptors.(ds), [GlobalSum()]), get_values.(get_energy.(ds))
    catch 
        false, 0.0, 0.0 
    end
    fd_flag, force_descriptors, forces = try  
        true, [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds], get_values.(get_forces.(ds))
    catch
        false, 0.0, 0.0
    end
    if d_flag & ~fd_flag 
        dim = length(descriptors[1])
        β = zeros(T, dim)

        p = UnivariateLinearProblem(descriptors, 
                energies, 
                β, 
                [1.0],
                Symmetric(zeros(dim, dim)))
    elseif ~d_flag & fd_flag 
        dim = length(force_descriptors[1][1])
        β = zeros(T, dim)

        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]
        p = UnivariateLinearProblem(force_descriptors,
            [reduce(vcat, fi) for fi in forces], 
            β, 
            [1.0], 
            Symmetric(zeros(dim, dim))
        )
        
    elseif d_flag & fd_flag 
        dim_d = length(descriptors[1])
        dim_fd = length(force_descriptors[1][1])

        if  (dim_d != dim_fd) 
            error("Descriptors and Force Descriptors have different dimension!") 
        else
            dim = dim_d
        end

        β = zeros(T, dim)
        forces =  [reduce(vcat, fi) for fi in forces]
        force_descriptors = [reduce(hcat, fi) for fi in force_descriptors]

        p = PotentialLearning.CovariateLinearProblem(energies,
                [reduce(vcat, fi) for fi in forces], 
                descriptors, 
                force_descriptors, 
                β, 
                [1.0], 
                [1.0], 
                Symmetric(zeros(dim, dim)))

    else 
        error("Either no (Energy, Descriptors) or (Forces, Force Descriptors) in DataSet")
    end
    p
end

